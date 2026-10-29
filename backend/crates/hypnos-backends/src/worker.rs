//! Thread-owned engine workers. A native context (llama.cpp's `llama_context`, for instance) is
//! neither `Sync` nor cheap to create, so each resident model gets one OS thread that owns the
//! context for the whole residency and serves requests sent over a channel. This module holds
//! the engine-agnostic plumbing: the handle that owns the thread, the shutdown protocol, and the
//! rule that jobs still queued when the worker stops are answered with an error instead of
//! being run on a model that is being released or dropped silently.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use hypnos_core::error::{HypnosError, Result};
use tracing::{debug, warn};

/// What the worker thread receives.
pub(crate) enum WorkerMsg<J> {
    Job(J),
    Shutdown,
}

/// A job that can be refused without running: the worker answers its caller with `error`.
pub(crate) trait FailJob {
    fn fail(self, error: HypnosError);
}

/// The worker thread's end of the channel plus the stop flag the handle raises on drop.
pub(crate) struct WorkerInbox<J> {
    rx: Receiver<WorkerMsg<J>>,
    stopping: Arc<AtomicBool>,
}

impl<J: FailJob> WorkerInbox<J> {
    /// Blocks for the next message. `None` means `timeout` passed with nothing to do (the worker
    /// can run housekeeping). Once the handle has asked for a stop, or the channel is closed,
    /// this returns `Shutdown` without running anything else that is queued.
    pub(crate) fn next(&self, timeout: Duration) -> Option<WorkerMsg<J>> {
        if self.stopping.load(Ordering::SeqCst) {
            return Some(WorkerMsg::Shutdown);
        }
        match self.rx.recv_timeout(timeout) {
            Ok(WorkerMsg::Job(job)) if self.stopping.load(Ordering::SeqCst) => {
                job.fail(HypnosError::BackendError(
                    "model is being released from VRAM".into(),
                ));
                Some(WorkerMsg::Shutdown)
            }
            Ok(msg) => Some(msg),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => Some(WorkerMsg::Shutdown),
        }
    }

    /// Non-blocking variant of [`next`](Self::next) for a worker that has work in progress:
    /// `None` means nothing is queued right now.
    pub(crate) fn try_next(&self) -> Option<WorkerMsg<J>> {
        if self.stopping.load(Ordering::SeqCst) {
            return Some(WorkerMsg::Shutdown);
        }
        match self.rx.try_recv() {
            Ok(msg) => Some(msg),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(WorkerMsg::Shutdown),
        }
    }

    /// Fails every job still queued after the worker decided to stop.
    pub(crate) fn drain_pending(&self, why: &str) {
        while let Ok(WorkerMsg::Job(job)) = self.rx.try_recv() {
            job.fail(HypnosError::BackendError(why.to_string()));
        }
    }
}

/// Owns a worker thread. Dropping the handle asks the thread to stop and waits for it, which is
/// how a model's context (and its VRAM) is released on demotion.
pub(crate) struct WorkerHandle<J: FailJob + Send + 'static> {
    name: String,
    tx: Sender<WorkerMsg<J>>,
    stopping: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl<J: FailJob + Send + 'static> WorkerHandle<J> {
    /// Starts `body` on a new thread named `name`. `body` receives the inbox and must return
    /// when it reads [`WorkerMsg::Shutdown`].
    pub(crate) fn spawn(
        name: &str,
        body: impl FnOnce(WorkerInbox<J>) + Send + 'static,
    ) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        let stopping = Arc::new(AtomicBool::new(false));
        let inbox = WorkerInbox {
            rx,
            stopping: stopping.clone(),
        };
        let join = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || body(inbox))
            .map_err(|e| {
                HypnosError::BackendError(format!("could not start worker thread: {e}"))
            })?;
        Ok(Self {
            name: name.to_string(),
            tx,
            stopping,
            join: Some(join),
        })
    }

    /// Queues `job`. If the worker is gone the job is failed immediately and an error returned.
    pub(crate) fn submit(&self, job: J) -> Result<()> {
        match self.tx.send(WorkerMsg::Job(job)) {
            Ok(()) => Ok(()),
            Err(mpsc::SendError(WorkerMsg::Job(job))) => {
                let msg = format!("engine worker '{}' is not running", self.name);
                job.fail(HypnosError::BackendError(msg.clone()));
                Err(HypnosError::BackendError(msg))
            }
            Err(mpsc::SendError(WorkerMsg::Shutdown)) => {
                unreachable!("submit never sends Shutdown")
            }
        }
    }
}

impl<J: FailJob + Send + 'static> Drop for WorkerHandle<J> {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::SeqCst);
        let _ = self.tx.send(WorkerMsg::Shutdown);
        if let Some(join) = self.join.take() {
            if join.join().is_err() {
                warn!(worker = %self.name, "engine worker thread panicked");
            } else {
                debug!(worker = %self.name, "engine worker stopped");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::Sender as StdSender;
    use std::time::Instant;

    struct Job {
        n: u32,
        reply: StdSender<Result<u32>>,
    }

    impl FailJob for Job {
        fn fail(self, error: HypnosError) {
            let _ = self.reply.send(Err(error));
        }
    }

    /// A worker that doubles numbers and sleeps on the first job, so later jobs queue up behind it.
    fn start(started: Option<StdSender<()>>) -> WorkerHandle<Job> {
        WorkerHandle::spawn("test-worker", move |inbox: WorkerInbox<Job>| {
            let mut first = true;
            loop {
                match inbox.next(Duration::from_millis(5)) {
                    None => continue,
                    Some(WorkerMsg::Shutdown) => break,
                    Some(WorkerMsg::Job(job)) => {
                        if first {
                            if let Some(tx) = &started {
                                let _ = tx.send(());
                                std::thread::sleep(Duration::from_millis(100));
                            }
                        }
                        first = false;
                        let _ = job.reply.send(Ok(job.n * 2));
                    }
                }
            }
            inbox.drain_pending("worker stopped");
        })
        .unwrap()
    }

    #[test]
    fn jobs_run_in_order_and_shutdown_joins() {
        let worker = start(None);
        let (tx, rx) = mpsc::channel();
        for n in 1..=3 {
            worker
                .submit(Job {
                    n,
                    reply: tx.clone(),
                })
                .unwrap();
        }
        let got: Vec<u32> = (0..3).map(|_| rx.recv().unwrap().unwrap()).collect();
        assert_eq!(got, vec![2, 4, 6]);
        let started = Instant::now();
        drop(worker);
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn pending_jobs_fail_cleanly_on_shutdown() {
        let (started_tx, started_rx) = mpsc::channel();
        let worker = start(Some(started_tx));
        let (tx, rx) = mpsc::channel();
        for n in 1..=3 {
            worker
                .submit(Job {
                    n,
                    reply: tx.clone(),
                })
                .unwrap();
        }
        // The stop request arrives while the first job is running; the two behind it must be
        // answered, not run against a model that is going away.
        started_rx.recv().unwrap();
        drop(worker);
        let first = rx.recv().unwrap();
        assert_eq!(first.unwrap(), 2);
        for _ in 0..2 {
            let err = rx.recv().unwrap().unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("worker stopped") || msg.contains("being released"),
                "{msg}"
            );
        }
    }
}

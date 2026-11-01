//! A response body that runs a callback when it is finished or dropped: the end of a request as
//! the client sees it, streams included (PERF-10 latency percentiles).

use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::{Body, Bytes};
use http_body::{Body as HttpBody, Frame, SizeHint};

pub struct DoneBody {
    inner: Body,
    on_done: Option<Box<dyn FnOnce() + Send>>,
}

impl DoneBody {
    pub fn wrap(inner: Body, on_done: impl FnOnce() + Send + 'static) -> Body {
        Body::new(DoneBody {
            inner,
            on_done: Some(Box::new(on_done)),
        })
    }

    fn finish(&mut self) {
        if let Some(f) = self.on_done.take() {
            f();
        }
    }
}

impl HttpBody for DoneBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let polled = Pin::new(&mut self.inner).poll_frame(cx);
        if let Poll::Ready(None) = polled {
            self.finish();
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for DoneBody {
    fn drop(&mut self) {
        // A client that went away mid-stream ends the request too.
        self.finish();
    }
}

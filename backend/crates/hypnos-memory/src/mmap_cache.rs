use hypnos_core::error::Result;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use tracing::{debug, warn};

/// Holds a memory-mapped model weight file (Tier 3 NVMe SSD) and manages OS page locking
/// to promote/demote pages into Tier 2 (Pinned Host RAM).
pub struct MmapWeightBuffer {
    mmap: Option<Mmap>,
    /// Fallback or synthetic byte length when testing without physical multi-GB weight files on disk
    len_bytes: usize,
    is_locked_in_ram: bool,
}

impl MmapWeightBuffer {
    /// Opens a weight file on NVMe SSD using `memmap2` (Tier 3 Cold state).
    /// If the file does not exist on disk yet, initializes a metadata-backed virtual handle.
    pub fn open_or_virtual(path: &Path, configured_bytes: u64) -> Result<Self> {
        if path.exists() {
            let file = File::open(path)?;
            let metadata_len = file.metadata()?.len() as usize;
            let mmap = unsafe { Mmap::map(&file)? };
            debug!(
                path = %path.display(),
                size_mb = metadata_len / (1024 * 1024),
                "Mapped model weight file on Tier 3 (NVMe SSD)"
            );
            Ok(Self {
                mmap: Some(mmap),
                len_bytes: metadata_len,
                is_locked_in_ram: false,
            })
        } else {
            debug!(
                path = %path.display(),
                configured_mb = configured_bytes / (1024 * 1024),
                "Weight file not yet present on disk; using configured byte footprint"
            );
            Ok(Self {
                mmap: None,
                len_bytes: configured_bytes as usize,
                is_locked_in_ram: false,
            })
        }
    }

    pub fn len_bytes(&self) -> u64 {
        self.len_bytes as u64
    }

    pub fn is_locked_in_ram(&self) -> bool {
        self.is_locked_in_ram
    }

    /// Raw pointer to the mapped host memory buffer (if backed by a physical file).
    /// Can be passed directly to `cudaHostRegister`, `hipHostRegister`, or `MTLDevice::newBufferWithBytesNoCopy`.
    pub fn host_ptr(&self) -> Option<usize> {
        self.mmap.as_ref().map(|m| m.as_ptr() as usize)
    }

    /// Promotes the mapped pages from Tier 3 (NVMe SSD) into Tier 2 (Pinned/Locked Host RAM).
    /// Touches every 4KB page to fault it into physical RAM and invokes `VirtualLock` / `mlock`.
    pub fn pin_in_host_ram(&mut self, lock_os_pages: bool) -> Result<()> {
        if self.is_locked_in_ram {
            return Ok(());
        }

        if let Some(ref mmap) = self.mmap {
            // 1. Pre-fault pages sequentially (page-warming) so NVMe reads at full sequential bandwidth
            let slice = mmap.as_ref();
            let mut checksum: u8 = 0;
            for byte in slice.iter().step_by(4096) {
                checksum = checksum.wrapping_add(*byte);
            }
            std::hint::black_box(checksum);

            // 2. Lock pages in physical host RAM so OS never swaps them out
            if lock_os_pages && !slice.is_empty() {
                Self::os_lock_memory(slice.as_ptr(), slice.len());
            }
        }

        self.is_locked_in_ram = true;
        Ok(())
    }

    /// Demotes the weight buffer from Tier 2 (Pinned Host RAM) back to Tier 3 (NVMe SSD)
    /// by unlocking the physical pages (`VirtualUnlock` / `munlock`).
    pub fn unpin_from_host_ram(&mut self) -> Result<()> {
        if !self.is_locked_in_ram {
            return Ok(());
        }

        if let Some(ref mmap) = self.mmap {
            let slice = mmap.as_ref();
            if !slice.is_empty() {
                Self::os_unlock_memory(slice.as_ptr(), slice.len());
            }
        }

        self.is_locked_in_ram = false;
        Ok(())
    }

    #[cfg(windows)]
    fn os_lock_memory(ptr: *const u8, len: usize) {
        use windows_sys::Win32::System::Memory::VirtualLock;
        let ok = unsafe { VirtualLock(ptr as *const _, len) };
        if ok == 0 {
            warn!(
                len_mb = len / (1024 * 1024),
                "VirtualLock hit OS working-set quota; pages remain warm in OS page cache"
            );
        }
    }

    #[cfg(windows)]
    fn os_unlock_memory(ptr: *const u8, len: usize) {
        use windows_sys::Win32::System::Memory::VirtualUnlock;
        unsafe {
            let _ = VirtualUnlock(ptr as *const _, len);
        }
    }

    #[cfg(unix)]
    fn os_lock_memory(ptr: *const u8, len: usize) {
        unsafe {
            let _ = libc::madvise(ptr as *mut _, len, libc::MADV_WILLNEED);
            if libc::mlock(ptr as *const _, len) != 0 {
                warn!(
                    len_mb = len / (1024 * 1024),
                    "mlock hit RLIMIT_MEMLOCK; pages remain warm in OS page cache"
                );
            }
        }
    }

    #[cfg(unix)]
    fn os_unlock_memory(ptr: *const u8, len: usize) {
        unsafe {
            let _ = libc::munlock(ptr as *const _, len);
            let _ = libc::madvise(ptr as *mut _, len, libc::MADV_DONTNEED);
        }
    }
}

/// Raises the process working-set limits so `VirtualLock` can pin up to `bytes` of model weights.
///
/// Windows only lets a process lock about as many pages as its *minimum* working set, so the
/// minimum (not just the maximum) has to cover the pinned budget. If the OS refuses the requested
/// size (not enough physical memory, missing privilege), the request is halved until it is accepted
/// so that at least part of the budget becomes lockable. On other platforms this is a no-op
/// (`mlock` is governed by `RLIMIT_MEMLOCK`).
#[cfg(windows)]
pub fn raise_working_set_limit(bytes: u64) {
    use windows_sys::Win32::System::Memory::{
        SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MAX_DISABLE,
        QUOTA_LIMITS_HARDWS_MIN_DISABLE,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let floor: usize = 512 * 1024 * 1024;
    let headroom: usize = 1024 * 1024 * 1024;
    let mut min: usize = (bytes as usize).max(floor);
    let mut last_err = 0u32;
    for _ in 0..8 {
        let max = min.saturating_add(headroom);
        let ok = unsafe {
            SetProcessWorkingSetSizeEx(
                GetCurrentProcess(),
                min,
                max,
                QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE,
            )
        };
        if ok != 0 {
            if (min as u64) < bytes {
                warn!(
                    requested_mb = bytes / (1024 * 1024),
                    granted_mb = min / (1024 * 1024),
                    "Only part of the pinned Tier 2 RAM budget can be locked; larger models stay in the page cache"
                );
            } else {
                debug!(
                    min_mb = min / (1024 * 1024),
                    max_mb = max / (1024 * 1024),
                    "Raised process working-set limits for pinned Tier 2 RAM"
                );
            }
            return;
        }
        last_err = unsafe { windows_sys::Win32::Foundation::GetLastError() };
        if min <= floor {
            break;
        }
        min = (min / 2).max(floor);
    }
    warn!(
        requested_mb = bytes / (1024 * 1024),
        win32_error = last_err,
        "Could not raise the working-set limit; VirtualLock will fall back to the OS page cache"
    );
}

#[cfg(not(windows))]
pub fn raise_working_set_limit(_bytes: u64) {}

impl Drop for MmapWeightBuffer {
    fn drop(&mut self) {
        let _ = self.unpin_from_host_ram();
    }
}

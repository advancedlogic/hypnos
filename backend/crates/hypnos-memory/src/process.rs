//! Host memory readings straight from the OS: installed physical RAM, and how much of it this
//! process occupies. Both are cheap enough to read before every model promotion.

/// Installed physical RAM in bytes, if the OS reports it.
pub fn physical_memory_bytes() -> Option<u64> {
    imp::physical_memory_bytes().filter(|&b| b > 0)
}

/// Physical memory this process occupies, as the OS accounts it:
/// - macOS: `phys_footprint` (what Activity Monitor shows; includes Metal allocations),
/// - Linux: resident set size (`VmRSS`),
/// - Windows: working set.
///
/// GPU memory on a discrete card is not part of any of these.
pub fn process_footprint_bytes() -> Option<u64> {
    imp::process_footprint_bytes().filter(|&b| b > 0)
}

#[cfg(target_os = "macos")]
mod imp {
    pub fn physical_memory_bytes() -> Option<u64> {
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let rc = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                &mut value as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0).then_some(value)
    }

    pub fn process_footprint_bytes() -> Option<u64> {
        let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            libc::proc_pid_rusage(
                std::process::id() as libc::c_int,
                libc::RUSAGE_INFO_V2,
                &mut info as *mut libc::rusage_info_v2 as *mut libc::rusage_info_t,
            )
        };
        (rc == 0).then_some(info.ri_phys_footprint)
    }
}

#[cfg(target_os = "linux")]
mod imp {
    /// Reads a `kB` field such as `MemTotal:  65843212 kB` from a /proc file.
    fn proc_kib(path: &str, field: &str) -> Option<u64> {
        let text = std::fs::read_to_string(path).ok()?;
        let line = text.lines().find(|l| l.starts_with(field))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    }

    pub fn physical_memory_bytes() -> Option<u64> {
        proc_kib("/proc/meminfo", "MemTotal:")
    }

    pub fn process_footprint_bytes() -> Option<u64> {
        proc_kib("/proc/self/status", "VmRSS:")
    }
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    pub fn physical_memory_bytes() -> Option<u64> {
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0).then_some(status.ullTotalPhys)
    }

    pub fn process_footprint_bytes() -> Option<u64> {
        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        let cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        counters.cb = cb;
        let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, cb) };
        (ok != 0).then_some(counters.WorkingSetSize as u64)
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
mod imp {
    pub fn physical_memory_bytes() -> Option<u64> {
        None
    }
    pub fn process_footprint_bytes() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn readings_are_plausible_on_supported_platforms() {
        if cfg!(any(target_os = "macos", target_os = "linux", windows)) {
            let physical = super::physical_memory_bytes().expect("physical RAM");
            let footprint = super::process_footprint_bytes().expect("process footprint");
            assert!(physical >= 1 << 30);
            assert!(footprint > 0 && footprint < physical);
        }
    }
}

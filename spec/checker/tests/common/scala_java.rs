//! Shared by the tests that run packaged Scala jars (`#[path]` module).
use std::path::{Path, PathBuf};
use std::process::Command;

/// The Java that runs the packaged Scala jars: rt.scala prints floats with
/// `Double.toString`, the shortest round trip only since Java 19 (10.5).
/// In order: `RCE_JAVA`, `$JAVA_HOME/bin/java`, `java` on the path, and
/// the JVM of `scala-cli --jvm 21`; the first that runs and is 19 or later.
/// On macOS without a JDK, `java` on the path is a stub that exits 1.
pub fn scala_java() -> Option<PathBuf> {
    static J: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    J.get_or_init(|| {
        let major = |j: &Path| -> Option<u32> {
            let o = Command::new(j).arg("-version").output().ok()?;
            if !o.status.success() {
                return None;
            }
            let t = String::from_utf8_lossy(&o.stderr);
            let v = t.split('"').nth(1)?;
            let m = v.split('.').next()?.parse().ok()?;
            Some(if m == 1 { v.split('.').nth(1)?.parse().ok()? } else { m })
        };
        let mut cands: Vec<PathBuf> = vec![];
        if let Ok(j) = std::env::var("RCE_JAVA") && !j.is_empty() {
            cands.push(j.into());
        }
        if let Ok(h) = std::env::var("JAVA_HOME") && !h.is_empty() {
            cands.push(Path::new(&h).join("bin/java"));
        }
        cands.push("java".into());
        if let Some(j) = cands.into_iter().find(|j| major(j).is_some_and(|m| m >= 19)) {
            return Some(j);
        }
        let d = std::env::temp_dir().join(format!("rce-fuel-jh-{}", std::process::id()));
        std::fs::create_dir_all(&d).ok()?;
        let sc = d.join("h.sc");
        std::fs::write(&sc, "println(System.getProperty(\"java.home\"))\n").ok()?;
        let o = Command::new("scala-cli").args(["run", "-q", "--server=false", "--jvm", "21"]).arg(&sc).output().ok()?;
        let h = String::from_utf8_lossy(&o.stdout).lines().last()?.trim().to_string();
        let j = Path::new(&h).join("bin/java");
        major(&j).is_some_and(|m| m >= 19).then_some(j)
    })
    .clone()
}

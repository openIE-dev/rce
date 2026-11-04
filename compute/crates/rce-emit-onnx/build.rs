fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A vendored protoc, so the build needs no system protobuf compiler.
    let mut config = prost_build::Config::new();
    config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(false)
        .compile_with_config(config, &["proto/onnx.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/onnx.proto");
    Ok(())
}

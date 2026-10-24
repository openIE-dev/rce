// The playground checker has the same built-in standard library as `rce`
// (specification 4.1): one generator, shared with spec/checker/build.rs.
include!("../checker/build_stdlib.rs");

fn main() {
    embed_stdlib();
}

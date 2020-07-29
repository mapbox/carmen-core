use ::test_utils::load_db_from_json;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        panic!("Expected 3 arguments: an input path, a splits file path, and a gridstore path")
    }
    load_db_from_json(&args[1], &args[2], &args[3]);
}

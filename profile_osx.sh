#!/bin/bash

# if current build profile doesn't include debug symbols, temporarily add
# config to include them
if [[ $(grep profile.release Cargo.toml | wc -l) -eq 0 ]]; then
    cp Cargo.toml Cargo.toml-$$
    printf "\n\n[profile.release]\ndebug = true" >> Cargo.toml
fi

# build the benchmarks
cargo bench --no-run $1
# find the build benchmark
BUILD=$(ls -t target/release/deps/benchmarks* | grep -v "\.d$" | head -n 1)
# run benchmark, modified from http://carol-nichols.com/2017/04/20/rust-profiling-with-dtrace-on-osx/
sudo -E dtrace -c "./$BUILD $1" -o out-$$.stacks -n 'profile-997 /pid == $target/ { @[ustack(100)] = count(); }'

if [ ! -d '/tmp/FlameGraph' ]; then
    git clone https://github.com/brendangregg/FlameGraph /tmp/FlameGraph
fi

/tmp/FlameGraph/stackcollapse.pl out-$$.stacks | /tmp/FlameGraph/flamegraph.pl > graph-$$.svg
open graph-$$.svg -a "/Applications/Google Chrome.app/"

# if we mangled Cargo.toml, put it back
if [ -f Cargo.toml-$$ ]; then
    mv Cargo.toml-$$ Cargo.toml
fi
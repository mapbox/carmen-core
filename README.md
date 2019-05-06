## carmen-core
Carmen-core is a backend storage for carmen written in rust.
NOTE: The node support is work-in-progress, currently you can't require this module in your project.

## Development

### Installation
The latest stable version of Rust and neon cli must be installed

To install rust, follow instructions at https://www.rust-lang.org/tools/install

To install `neon-cli` run:

```
npm install --global neon-cli
# OR
yarn global add neon-cli
```

`carmen-core` requires node-gyp, which [only works with python 2](https://github.com/nodejs/node-gyp/issues/1337). If you use [pyenv](https://github.com/pyenv/pyenv) to manage python versions, the `.python-version ` file checked into this repo should be sufficient. Otherwise, try the instructions in [this comment](https://github.com/nodejs/node-gyp/issues/1337#issuecomment-370135532):

```
brew install python@2
# follow the instructions printed after installing `python@2` to get python 2 on your PATH 
# e.g. echo 'export PATH="/usr/local/opt/python@2/bin:$PATH"' >> ~/.bashrc
npm config set python python2.7
```


To install `carmen-core` run:

```
npm install
```

To build from source run:

```
neon build
```

### Test

To run tests for the rust test-suite, run the following commands:

```
cd rust-src
cargo test
```

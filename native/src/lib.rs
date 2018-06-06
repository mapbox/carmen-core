#[macro_use]
extern crate neon;
extern crate fuzzy_phrase;

use std::path::Path;

use neon::mem::Handle;
use neon::vm::{This, FunctionCall, JsResult};
use neon::js::{JsFunction, Object, JsString, Value};
use neon::js::class::{JsClass, Class};

use fuzzy_phrase::glue::{FuzzyPhraseSetBuilder, FuzzyPhraseSet};

// check argument type: https://github.com/Brooooooklyn/sourcemap-decoder/blob/master/native/src/lib.rs#L21-L29
trait CheckArgument {
  fn check_argument<V: Value>(&mut self, i: i32) -> JsResult<V>;
}

impl<'a, T: This> CheckArgument for FunctionCall<'a, T> {
  fn check_argument<V: Value>(&mut self, i: i32) -> JsResult<V> {
    self.arguments.require(self.scope, i)?.check::<V>()
  }
}

impl AsRef<Path> for String {
    fn as_ref() -> &Path {
        
    }
}
// FuzzyPhraseSetBuilder::new!(fuzzy_phrase::FuzzyPhraseSetBuilder);

declare_types! {
    pub class JsFuzzyPhraseSetBuilder as JsFuzzyPhraseSetBuilder for Option<FuzzyPhraseSetBuilder> {
        init(mut call) {
            let filename = call
                .check_argument::<JsString>(0)
                ?.value();
            let path = filename.as_ref();
            let mut build = FuzzyPhraseSetBuilder::new(path).unwrap();
            Ok(Some(build))
        }
    }

    // pub class JsFuzzyPhraseSet as JsFuzzyPhraseSet for Option<FuzzyPhraseSet<std::string::String>> {
    //     init(mut call) {
    //         let filename = call
    //             .check_argument::<JsString>(0)
    //             ?.value();
    //         let set = unsafe { FuzzyPhraseSet::from_path(filename).unwrap() };
    //         Ok(set)
    //     }
    // }
}

register_module!(m, {

    let class: Handle<JsClass<JsFuzzyPhraseSetBuilder>> = try!(JsFuzzyPhraseSetBuilder::class(m.scope));
    let constructor: Handle<JsFunction<JsFuzzyPhraseSetBuilder>> = try!(class.constructor(m.scope));
    try!(m.exports.set("FuzzyPhraseSetBuilder", constructor));

    Ok(())
});

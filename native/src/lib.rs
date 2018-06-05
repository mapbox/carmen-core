#[macro_use]
extern crate neon;
extern crate fuzzy_phrase;

use neon::vm::{Call, JsResult, FunctionCall, This};
use neon::js::{Value};
use neon::js::JsString;

use fuzzy_phrase::{FuzzyPhraseSetBuilder};

// check argument type
// trait CheckArgument {
//   fn check_argument<V: Value>(&mut self, i: i32) -> JsResult<V>;
// }
//
// impl<'a, T: This> CheckArgument for FunctionCall<'a, T> {
//   fn check_argument<V: Value>(&mut self, i: i32) -> JsResult<V> {
//     self.arguments.require(self.scope, i)?.check::<V>()
//   }
// }

declare_types! {
    pub class JsFuzzyPhraseSetBuilder as JsFuzzyPhraseSetBuilder for Option<FuzzyPhraseSetBuilder<P: AsRef<Path>>(path: P) -> Result<Self, Box<Error>>> {
        init(mut call) {

            Ok(())
        }
    }
}

register_module!(m, {


});

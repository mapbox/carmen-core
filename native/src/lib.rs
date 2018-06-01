#[macro_use]
extern crate neon;
extern crate fuzzy_phrase;

use neon::vm::{Call, JsResult};
use neon::js::JsString;

trait CheckArgument {
  fn check_argument<V: Value>(&mut self, i: i32) -> JsResult<V>;
}

impl<'a, T: This> CheckArgument for FunctionCall<'a, T> {
  fn check_argument<V: Value>(&mut self, i: i32) -> JsResult<V> {
    self.arguments.require(self.scope, i)?.check::<V>()
  }
}

declare_types! {
    // pub class JsFuzzyPhraseSetBuilder as JsFuzzyPhraseSetBuilder for Option<FuzzyPhraseSetBuilder<io::BufWriter<File>>>{
    //
    // }
}

register_module!(m, {


});

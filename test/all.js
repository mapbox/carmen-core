const fuzzy = require('../lib');
const assert = require('assert');
const tape = require('tape');

tape('build FuzzyPhraseSetBuilder', (t) => {
    let build = new fuzzy.FuzzyPhraseSetBuilder("set.fuzzy");
    t.ok(build, "FuzzyPhraseSetBuilder built");
    t.throws(() => { new fuzzy.FuzzyPhraseSetBuilder()});
    t.throws(() => { new fuzzy.FuzzyPhraseSetBuilder("/etc/passwd")});
    t.throws(() => { new fuzzy.FuzzyPhraseSetBuilder({})});
    t.throws(() => { new fuzzy.FuzzyPhraseSetBuilder(7)});
    t.end();
})

tape("FuzzyPhraseSetBuilder insertion and Set lookup", (t) => {
    let build = new fuzzy.FuzzyPhraseSetBuilder("set.fuzzy");
    // ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]

    build.insert(["bruce"]);
    build.insert(["clarence"]);
    build.insert(["stevie"]);
    // build.insert({});
    build.insert(["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"])

    build.finish();

    let set = new fuzzy.FuzzyPhraseSet("set.fuzzy");
    set.contains(["bruce", "clarence", "stevie"]);
    set.contains(["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]);
    t.notOk(set.contains(["FOOBAR"]));

    t.throws(() => { new fuzzy.FuzzyPhraseSet() });
    t.throws(() => { new fuzzy.FuzzyPhraseSet("/etc/passwd") });
    t.throws(() => { new fuzzy.FuzzyPhraseSet({}) });
    t.throws(() => { new fuzzy.FuzzyPhraseSet(7) });
    t.end();
})

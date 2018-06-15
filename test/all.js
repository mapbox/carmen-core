const fuzzy = require('../lib');
const assert = require('assert');
const tape = require('tape');

tape('build SetBuilder', (t) => {
    let build = new fst.SetBuilder("set.fst");
    t.ok(build, "SetBuilder built");
    t.throws(() => { new fst.SetBuilder()});
    t.throws(() => { new fst.SetBuilder("/etc/passwd")});
    t.throws(() => { new fst.SetBuilder({})});
    t.throws(() => { new fst.SetBuilder(7)});
    t.end();
})

tape("SetBuilder insertion and Set lookup", (t) => {
    let build = new fst.SetBuilder("set.fst");

    build.insert("bruce");
    build.insert("clarence");
    build.insert("stevie");

    build.finish();

    let set = new fst.Set("set.fst");
    for (let name of ["bruce", "clarence", "stevie"]) {
        t.ok(set.contains(name), "Set contains expected word");
    }

    t.throws(() => { new fst.Set() });
    t.throws(() => { new fst.Set("/etc/passwd") });
    t.throws(() => { new fst.Set({}) });
    t.throws(() => { new fst.Set(7) });
    t.end();
})

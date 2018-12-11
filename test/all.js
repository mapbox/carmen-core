'use strict';

const tape = require('tape');
const tmp = require('tmp');
const rimraf = require('rimraf').sync;
const fs = require('fs');
const fuzzy = require('../lib');

const tmpDir = tmp.dirSync();

tape('build FuzzyPhraseSetBuilder', (t) => {
    const builder = new fuzzy.FuzzyPhraseSetBuilder(tmpDir.name);
    t.ok(builder, 'FuzzyPhraseSetBuilder built');
    t.throws(() => {
        new fuzzy.FuzzyPhraseSetBuilder();
    }, 'throws on not enough arguments');
    t.throws(() => {
        new fuzzy.FuzzyPhraseSetBuilder('/etc/passwd');
    }, 'throws on not a directory');
    t.throws(() => {
        new fuzzy.FuzzyPhraseSetBuilder({});
    }, 'throws on wrong type argument');
    t.throws(() => {
        new fuzzy.FuzzyPhraseSetBuilder(7);
    }, 'throws on wrong type arguments');
    t.end();
});

tape('FuzzyPhraseSetBuilder insertion and Set lookup', (t) => {
    const builder = new fuzzy.FuzzyPhraseSetBuilder(tmpDir.name);

    builder.insert(['100', 'main', 'street']);
    builder.insert(['200', 'main', 'street']);
    builder.insert(['100', 'main', 'ave']);
    builder.insert(['300', 'mlk', 'blvd']);
    builder.finish();

    const set = new fuzzy.FuzzyPhraseSet(tmpDir.name);
    t.ok(set.contains(['100', 'main', 'street']), 'FuzzyPhraseSet contains()');
    t.ok(set.contains(['200', 'main', 'street']), 'FuzzyPhraseSet contains()');
    t.ok(set.contains(['100', 'main', 'ave']), 'FuzzyPhraseSet contains()');
    t.ok(set.contains(['300', 'mlk', 'blvd']), 'FuzzyPhraseSet contains()');

    t.ok(set.contains(['100', 'main', 'street']), 'FuzzyPhraseSet contains()');
    const phraseStatic = ['100', 'main', 'street'];
    t.ok(set.contains(phraseStatic), 'FuzzyPhraseSet contains()');
    const phraseVec = ['100', 'main', 'street'];
    t.ok(set.contains(phraseVec), 'FuzzyPhraseSet contains()');

    t.notOk(set.contains(['x']), 'FuzzyPhraseSet does not contains()');
    t.notOk(set.contains(['100', 'main']), 'FuzzyPhraseSet does not contains()');
    t.notOk(
        set.contains(['100', 'main', 's']),
        'FuzzyPhraseSet does not contains()'
    );
    t.notOk(
        set.contains(['100', 'main', 'streetr']),
        'FuzzyPhraseSet does not contains()'
    );
    t.notOk(
        set.contains(['100', 'main', 'street', 'r']),
        'FuzzyPhraseSet does not contains()'
    );
    t.notOk(
        set.contains(['100', 'main', 'street', 'ave']),
        'FuzzyPhraseSet does not contains()'
    );

    t.ok(
        set.containsPrefix(['100', 'main', 'street']),
        'FuzzyPhraseSet containsPrefix()'
    );
    t.ok(
        set.containsPrefix(['200', 'main', 'street']),
        'FuzzyPhraseSet containsPrefix()'
    );
    t.ok(
        set.containsPrefix(['100', 'main', 'ave']),
        'FuzzyPhraseSet containsPrefix()'
    );
    t.ok(
        set.containsPrefix(['300', 'mlk', 'blvd']),
        'FuzzyPhraseSet containsPrefix()'
    );

    t.ok(set.containsPrefix(['100', 'main']), 'FuzzyPhraseSet containsPrefix()');
    t.ok(set.containsPrefix(['200', 'main']), 'FuzzyPhraseSet containsPrefix()');
    t.ok(set.containsPrefix(['100', 'main']), 'FuzzyPhraseSet containsPrefix()');
    t.ok(set.containsPrefix(['300', 'mlk']), 'FuzzyPhraseSet containsPrefix()');

    t.notOk(
        set.containsPrefix(['100', 'man']),
        'FuzzyPhraseSet does not containsPrefix()'
    );
    t.notOk(
        set.containsPrefix(['400', 'main']),
        'FuzzyPhraseSet does not containsPrefix()'
    );
    t.notOk(
        set.containsPrefix(['100', 'main', 'street', 'x']),
        'FuzzyPhraseSet does not containsPrefix()'
    );

    t.ok(
        set.fuzzyMatch(['100', 'man', 'street'], 1, 1),
        'FuzzyPhraseSet fuzzyMatch()'
    );

    t.deepEquals(
        set.fuzzyMatch(['100', 'man', 'street'], 1, 1),
        [{ edit_distance: 1, phrase: ['100', 'main', 'street'] }],
        'FuzzyPhraseSet fuzzyMatch()'
    );

    t.deepEquals(
        set.fuzzyMatch(['100', 'man', 'stret'], 1, 2),
        [{ edit_distance: 2, phrase: ['100', 'main', 'street'] }],
        'FuzzyPhraseSet fuzzyMatch()'
    );

    t.deepEquals(
        set.fuzzyMatchPrefix(['100', 'man'], 1, 1),
        [{ phrase: ['100', 'main'], edit_distance: 1 }],
        'FuzzyPhraseSet fuzzyMatchPrefix()'
    );

    t.deepEquals(
        set.fuzzyMatchPrefix(['100', 'man', 'str'], 1, 1),
        [{ phrase: ['100', 'main', 'str'], edit_distance: 1 }],
        'FuzzyPhraseSet fuzzyMatchPrefix()'
    );

    t.deepEquals(
        set.fuzzyMatchMulti(
            [
                [['100', 'man', 'street'], false],
                [['100', 'man', 'stret'], false],
                [['100', 'man'], true],
                [['100', 'man', 'str'], true]
            ],
            1,
            2
        ),
        [
            [{ edit_distance: 1, phrase: ['100', 'main', 'street'] }],
            [{ edit_distance: 2, phrase: ['100', 'main', 'street'] }],
            [{ phrase: ['100', 'main'], edit_distance: 1 }],
            [{ phrase: ['100', 'main', 'str'], edit_distance: 1 }]
        ],
        'FuzzyPhraseSet fuzzyMatchMulti()'
    );

    t.deepEquals(
        set.fuzzyMatchWindows(
            '100 man street washington 200'.split(' '),
            0,
            0,
            false
        ),
        []
    );

    t.deepEquals(
        set.fuzzyMatchWindows(
            '100 man street washington 200'.split(' '),
            1,
            1,
            false
        ),
        [
            {
                edit_distance: 1,
                ends_in_prefix: false,
                phrase: ['100', 'main', 'street'],
                start_position: 0
            }
        ]
    );

    t.deepEquals(
        set.fuzzyMatchWindows(
            '100 man street washington 200'.split(' '),
            1,
            1,
            true
        ),
        [
            {
                edit_distance: 1,
                ends_in_prefix: false,
                phrase: ['100', 'main', 'street'],
                start_position: 0
            },
            {
                edit_distance: 0,
                ends_in_prefix: true,
                phrase: ['200'],
                start_position: 4
            }
        ]
    );

    rimraf(tmpDir.name);

    t.end();
});

tape('load word replacements', (t) => {
    let builder = new fuzzy.FuzzyPhraseSetBuilder(tmpDir.name);
    builder.loadWordReplacements([{ from: 'Street', to: 'Str' }]);
    builder.finish();
    var word_replacements_obj = JSON.parse(fs.readFileSync(tmpDir.name + '/metadata.json')).word_replacements;
    t.deepEquals(word_replacements_obj, [ { from: 'Street', to: 'Str' } ], 'ok, loads word replacements')
    t.end();
});

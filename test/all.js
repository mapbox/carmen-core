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
    t.ok(set.contains(['100', 'main', 'street'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');
    t.ok(set.contains(['200', 'main', 'street'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');
    t.ok(set.contains(['100', 'main', 'ave'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');
    t.ok(set.contains(['300', 'mlk', 'blvd'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');

    t.ok(set.contains(['100', 'main', 'street'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');
    const phraseStatic = ['100', 'main', 'street'];
    t.ok(set.contains(phraseStatic, fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');
    const phraseVec = ['100', 'main', 'street'];
    t.ok(set.contains(phraseVec, fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');

    t.notOk(set.contains(['x'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet does not contains()');
    t.notOk(set.contains(['100', 'main'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet does not contains()');
    t.notOk(
        set.contains(['100', 'main', 's'], fuzzy.ENDING_TYPE.nonPrefix),
        'FuzzyPhraseSet does not contains()'
    );
    t.notOk(
        set.contains(['100', 'main', 'streetr'], fuzzy.ENDING_TYPE.nonPrefix),
        'FuzzyPhraseSet does not contains()'
    );
    t.notOk(
        set.contains(['100', 'main', 'street', 'r'], fuzzy.ENDING_TYPE.nonPrefix),
        'FuzzyPhraseSet does not contains()'
    );
    t.notOk(
        set.contains(['100', 'main', 'street', 'ave'], fuzzy.ENDING_TYPE.nonPrefix),
        'FuzzyPhraseSet does not contains()'
    );

    t.ok(
        set.contains(['100', 'main', 'street'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet containsPrefix()'
    );
    t.ok(
        set.contains(['200', 'main', 'street'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet containsPrefix()'
    );
    t.ok(
        set.contains(['100', 'main', 'ave'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet containsPrefix()'
    );
    t.ok(
        set.contains(['300', 'mlk', 'blvd'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet containsPrefix()'
    );

    t.ok(set.contains(['100', 'main'], fuzzy.ENDING_TYPE.anyPrefix), 'FuzzyPhraseSet containsPrefix()');
    t.ok(set.contains(['200', 'main'], fuzzy.ENDING_TYPE.anyPrefix), 'FuzzyPhraseSet containsPrefix()');
    t.ok(set.contains(['100', 'main'], fuzzy.ENDING_TYPE.anyPrefix), 'FuzzyPhraseSet containsPrefix()');
    t.ok(set.contains(['300', 'mlk'], fuzzy.ENDING_TYPE.anyPrefix), 'FuzzyPhraseSet containsPrefix()');

    t.notOk(
        set.contains(['100', 'man'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet does not containsPrefix()'
    );
    t.notOk(
        set.contains(['400', 'main'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet does not containsPrefix()'
    );
    t.notOk(
        set.contains(['100', 'main', 'street', 'x'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet does not containsPrefix()'
    );

    t.ok(
        set.fuzzyMatch(['100', 'man', 'street'], 1, 1, fuzzy.ENDING_TYPE.nonPrefix),
        'FuzzyPhraseSet fuzzyMatch(..., fuzzy.ENDING_TYPE.nonPrefix)'
    );

    t.deepEquals(
        set.fuzzyMatch(['100', 'man', 'street'], 1, 1, fuzzy.ENDING_TYPE.nonPrefix),
        [{ edit_distance: 1, phrase: ['100', 'main', 'street'], ending_type: fuzzy.ENDING_TYPE.nonPrefix }],
        'FuzzyPhraseSet fuzzyMatch(..., fuzzy.ENDING_TYPE.nonPrefix)'
    );

    t.deepEquals(
        set.fuzzyMatch(['100', 'man', 'stret'], 1, 2, fuzzy.ENDING_TYPE.nonPrefix),
        [{ edit_distance: 2, phrase: ['100', 'main', 'street'], ending_type: fuzzy.ENDING_TYPE.nonPrefix }],
        'FuzzyPhraseSet fuzzyMatch(..., fuzzy.ENDING_TYPE.nonPrefix)'
    );

    t.deepEquals(
        set.fuzzyMatch(['100', 'man'], 1, 1, fuzzy.ENDING_TYPE.anyPrefix),
        [{ phrase: ['100', 'main'], edit_distance: 1, ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix }],
        'FuzzyPhraseSet fuzzyMatch(..., fuzzy.ENDING_TYPE.anyPrefix)'
    );

    t.deepEquals(
        set.fuzzyMatch(['100', 'man', 'str'], 1, 1, fuzzy.ENDING_TYPE.anyPrefix),
        [{ phrase: ['100', 'main', 'str'], edit_distance: 1, ending_type: fuzzy.ENDING_TYPE.anyPrefix }],
        'FuzzyPhraseSet fuzzyMatch(..., fuzzy.ENDING_TYPE.anyPrefix)'
    );

    t.deepEquals(
        set.fuzzyMatchMulti(
            [
                [['100', 'man', 'street'], fuzzy.ENDING_TYPE.nonPrefix],
                [['100', 'man', 'stret'], fuzzy.ENDING_TYPE.nonPrefix],
                [['100', 'man'], fuzzy.ENDING_TYPE.anyPrefix],
                [['100', 'man', 'str'], fuzzy.ENDING_TYPE.anyPrefix]
            ],
            1,
            2
        ),
        [
            [{ edit_distance: 1, phrase: ['100', 'main', 'street'], ending_type: fuzzy.ENDING_TYPE.nonPrefix }],
            [{ edit_distance: 2, phrase: ['100', 'main', 'street'], ending_type: fuzzy.ENDING_TYPE.nonPrefix }],
            [{ phrase: ['100', 'main'], edit_distance: 1, ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix }],
            [{ phrase: ['100', 'main', 'str'], edit_distance: 1, ending_type: fuzzy.ENDING_TYPE.anyPrefix }]
        ],
        'FuzzyPhraseSet fuzzyMatchMulti()'
    );

    t.deepEquals(
        set.fuzzyMatchWindows(
            '100 man street washington 200'.split(' '),
            0,
            0,
            fuzzy.ENDING_TYPE.nonPrefix
        ),
        []
    );

    t.deepEquals(
        set.fuzzyMatchWindows(
            '100 man street washington 200'.split(' '),
            1,
            1,
            fuzzy.ENDING_TYPE.nonPrefix
        ),
        [
            {
                edit_distance: 1,
                ending_type: fuzzy.ENDING_TYPE.nonPrefix,
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
            fuzzy.ENDING_TYPE.anyPrefix
        ),
        [
            {
                edit_distance: 1,
                ending_type: fuzzy.ENDING_TYPE.nonPrefix,
                phrase: ['100', 'main', 'street'],
                start_position: 0
            },
            {
                edit_distance: 0,
                ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix,
                phrase: ['200'],
                start_position: 4
            }
        ]
    );

    rimraf(tmpDir.name);

    t.end();
});

tape('load word replacements', (t) => {
    const builder = new fuzzy.FuzzyPhraseSetBuilder(tmpDir.name);
    builder.loadWordReplacements([{ from: 'Street', to: 'Str' }]);
    builder.finish();
    const word_replacements_obj = JSON.parse(fs.readFileSync(tmpDir.name + '/metadata.json')).word_replacements;
    t.deepEquals(word_replacements_obj, [{ from: 'Street', to: 'Str' }], 'ok, loads word replacements');
    t.end();
});

var addon = require('./native');

// wire up iterator creation from the JS side
addon.GridStore.prototype.keys = function() {
    const out = {};
    out[Symbol.iterator] = () => new addon.GridStoreKeyIterator(this);
    return out;
}

addon.ENDING_TYPE = {
    nonPrefix: 0,
    anyPrefix: 1,
    wordBoundaryPrefix: 2,
}

module.exports = addon;

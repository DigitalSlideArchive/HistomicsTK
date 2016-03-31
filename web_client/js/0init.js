/**
 * Top level namespace.
 */
var histomicstk = window.histomicstk || {};

_.extend(histomicstk, {
    models: {},
    collections: {},
    views: {},
    router: new Backbone.Router(),
    events: _.clone(Backbone.Events)
});

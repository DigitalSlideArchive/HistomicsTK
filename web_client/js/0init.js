/**
 * Top level namespace.
 */
var histomicstk = window.histomicstk || {};

_.extend(histomicstk, {
    models: {},
    collections: {},
    views: {},
    router: new Backbone.Router({
        routes: {
            ':gui': 'gui',
            '': 'main'
        }
    }),
    events: _.clone(Backbone.Events)
});

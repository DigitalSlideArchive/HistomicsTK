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

(function () {
    // Set histomicstk.rootPath which will serve as the path root
    // for all file dialogs.  This defaults to `/collections/TCGA`
    // but falls back to the logged in user model on error.
    var RootModel = girder.Model.extend({
        save: function () { return this; },
        fetch: function () {
            girder.restRequest({
                path: '/resource/lookup',
                data: {
                    path: '/collection/TCGA'
                }
            }).done(_.bind(function (model) {
                this.resourceName = 'collection';
                this.set(model);
            }, this)).error(_.bind(function () {
                girder.fetchCurrentUser().done(_.bind(function (user) {
                    this.resourceName = 'user';
                    if (user) {
                        this.set(user);
                    }
                    girder.events.on('g:login', function () {
                        histomicstk.rootPath.set(girder.currentUser.attributes);
                    });
                }, this));
            }, this));
            return this;
        }
    });

    if (!histomicstk.rootPath) {
        histomicstk.rootPath = new RootModel().fetch();
    }
})();

/**
 * Top level namespace.
 */
var histomicstk = window.histomicstk || {};

_.extend(histomicstk, {
    models: {},
    collections: {},
    views: {},
    router: new (girder.Router.extend({
        setQuery: function (name, value, options) {
            var curRoute = Backbone.history.fragment,
                routeParts = girder.dialogs.splitRoute(curRoute),
                queryString = girder.parseQueryString(routeParts.name);
            if (value === undefined || value === null) {
                delete queryString[name];
            } else {
                queryString[name] = value;
            }
            var unparsedQueryString = $.param(queryString);
            if (unparsedQueryString.length > 0) {
                unparsedQueryString = '?' + unparsedQueryString;
            }
            this._lastQueryString = queryString;
            this.navigate(routeParts.base + unparsedQueryString, options);
        },
        getQuery: function (name) {
            return (this._lastQueryString || {})[name];
        },
        execute: function (callback, args) {
            var query = girder.parseQueryString(args.pop());
            args.push(query);
            if (callback) {
                callback.apply(this, args);
            }

            _.each(this._lastQueryString || {}, function (value, key) {
                if (!_.has(query, key)) {
                    histomicstk.events.trigger('query:' + key, null, query);
                }
            });
            _.each(query, function (value, key) {
                histomicstk.events.trigger('query:' + key, value, query);
            });
            histomicstk.events.trigger('query', query);
            this._lastQueryString = query;
        }
    }))(),
    events: _.clone(Backbone.Events),
    dialogs: {
        login: new girder.views.LoginView({parentView: null}),
        register: new girder.views.RegisterView({parentView: null})
    }
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

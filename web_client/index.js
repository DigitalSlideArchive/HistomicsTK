import App from './app.js';
import * as views from './views';
import * as dialogs from './dialogs';
import router from './router';
import events from './events';

export {
    App,
    views,
    dialogs,
    router,
    events
};

/*
_.extend(histomicstk, {
    models: models,
    collections: collections,
    views: views,
    events: events,
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
                    events.trigger('query:' + key, null, query);
                }
            });
            _.each(query, function (value, key) {
                events.trigger('query:' + key, value, query);
            });
            events.trigger('query', query);
            this._lastQueryString = query;
        }
    }))(),
    dialogs: {
        login: new girder.views.LoginView({parentView: null}),
        register: new girder.views.RegisterView({parentView: null})
    }
});

(function () {
    // Set slicer.rootPath which will serve as the path root
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
                        // slicer.rootPath.set(girder.currentUser.attributes);
                    });
                }, this));
            }, this));
            return this;
        }
    });

    if (!slicer.rootPath) {
        slicer.rootPath = new RootModel().fetch();
    }
})();
*/

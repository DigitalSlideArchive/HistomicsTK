import _ from 'underscore';
import Backbone from 'backbone';
import { splitRoute, parseQueryString } from 'girder/misc';

import events from './events';

var router = new Backbone.Router();

router.setQuery = function setQuery(name, value, options) {
    var curRoute = Backbone.history.fragment,
        routeParts = splitRoute(curRoute),
        queryString = parseQueryString(routeParts.name);
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
};

router.getQuery = function getQuery(name) {
    return (this._lastQueryString || {})[name];
};

router.execute = function execute(callback, args) {
    var query = parseQueryString(args.pop());
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
};

export default router;

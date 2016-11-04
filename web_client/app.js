import Backbone from 'backbone';

import GirderApp from 'girder/views/App';
import eventStream from 'girder/utilities/EventStream';
import { getCurrentUser } from 'girder/auth';
import { splitRoute } from 'girder/misc';

import router from './router';
import HeaderView from './views/layout/HeaderView';

import layoutTemplate from './templates/layout/layout.pug';
import './stylesheets/layout/layout.styl';

var App = GirderApp.extend({

    render: function () {
        this.$el.html(layoutTemplate());

        new HeaderView({
            el: this.$('#g-app-header-container'),
            parentView: this
        }).render();

        return this;
    },

    /**
     * On login we re-render the current body view; whereas on
     * logout, we redirect to the front page.
     */
    login: function () {
        var route = splitRoute(Backbone.history.fragment).base;
        Backbone.history.fragment = null;
        eventStream.close();

        if (getCurrentUser()) {
            eventStream.open();
            router.navigate(route, {trigger: true});
        } else {
            router.navigate('/', {trigger: true});
        }
    }
});

export default App;

import Backbone from 'backbone';

import GirderApp from 'girder/views/App';
import eventStream from 'girder/utilities/EventStream';
import { getCurrentUser } from 'girder/auth';
import { splitRoute } from 'girder/misc';

import router from './router';
import HeaderView from './views/layout/HeaderView';
import bindRoutes from './routes';

import layoutTemplate from './templates/layout/layout.pug';
import './stylesheets/layout/layout.styl';

var App = GirderApp.extend({
    initialize(settings) {
        this.settings = settings;
        return GirderApp.prototype.initialize.apply(this, arguments);
    },

    render() {
        this.$el.html(layoutTemplate());

        this.histomicsHeader = new HeaderView({
            el: this.$('#g-app-header-container'),
            parentView: this,
            settings: this.settings
        }).render();

        return this;
    },

    /**
     * On login we re-render the current body view; whereas on
     * logout, we redirect to the front page.
     */
    login() {
        var route = splitRoute(Backbone.history.fragment).base;
        Backbone.history.fragment = null;
        eventStream.close();

        if (getCurrentUser()) {
            eventStream.open();
            router.navigate(route, {trigger: true});
        } else {
            router.navigate('/', {trigger: true});
        }
    },

    navigateTo(view) {
        if (this.bodyView instanceof view) {
            return this;
        }
        return GirderApp.prototype.navigateTo.apply(this, arguments);
    },

    bindRoutes: bindRoutes
});

export default App;

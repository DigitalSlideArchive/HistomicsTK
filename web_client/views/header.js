import LayoutHeaderUserView from 'girder/views/layout/LayoutHeaderUserView';
import { logout } from 'girder/auth';

import router from '../router';

import header from '../templates/header.pug';
import '../stylesheets/header.styl';

var Header = LayoutHeaderUserView.extend({
    events: {
        'click a.g-analysis-item': '_selectAnalysis',
        'click .g-register': '_register',
        'click .g-login': '_login',
        'click .g-logout': '_logout',
        'click .g-open-image': '_openImage'
    },
    initialize: function () {
        this.analyses = [];
        girder.restRequest({path: 'HistomicsTK/HistomicsTK/docker_image', error: null})
            .then(_.bind(function (data) {
                this.analyses = data;
                this.render();
            }, this));

        this.listenTo(girder.events, 'g:login', this.render);
        this.listenTo(girder.events, 'g:login-changed', this.render);
        this.listenTo(girder.events, 'g:logout', this.render);
    },
    render: function () {
        this.$el.html(header({
            user: girder.currentUser,
            analyses: this.analyses
        }));
        this.$('[data-submenu]').submenupicker();
        return this;
    },
    _selectAnalysis: function (evt) {
        var $el = $(evt.target);
        router.setQuery('analysis', $el.data('api'), {trigger: true});
        evt.preventDefault();
    },
    _register: function (evt) {
        router.setQuery('dialog', 'register', {trigger: true});
        evt.preventDefault();
    },
    _login: function (evt) {
        router.setQuery('dialog', 'login', {trigger: true});
        evt.preventDefault();
    },
    _logout: function (evt) {
        logout();
        evt.preventDefault();
    },
    _openImage: function (evt) {
        router.setQuery('dialog', 'image', {trigger: true});
        evt.preventDefault();
    }
});

export default Header;

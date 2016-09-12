histomicstk.views.Header = girder.views.LayoutHeaderUserView.extend({
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
        this.$el.html(histomicstk.templates.header({
            user: girder.currentUser,
            analyses: this.analyses
        }));
        this.$('[data-submenu]').submenupicker();
        return this;
    },
    _selectAnalysis: function (evt) {
        var $el = $(evt.target);
        histomicstk.router.setQuery('analysis', $el.data('api'), {trigger: true});
        evt.preventDefault();
    },
    _register: function (evt) {
        histomicstk.router.setQuery('dialog', 'register', {trigger: true});
        evt.preventDefault();
    },
    _login: function (evt) {
        histomicstk.router.setQuery('dialog', 'login', {trigger: true});
        evt.preventDefault();
    },
    _logout: function (evt) {
        girder.logout();
        evt.preventDefault();
    },
    _openImage: function (evt) {
        histomicstk.router.setQuery('dialog', 'image', {trigger: true});
        evt.preventDefault();
    }
});

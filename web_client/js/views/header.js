histomicstk.views.Header = girder.views.LayoutHeaderUserView.extend({
    events: {
        'click #g-analysis-menu a': '_selectAnalysis',
        'click .g-register': '_register',
        'click .g-login': '_login',
        'click .g-logout': '_logout'
    },
    initialize: function () {
        this.analyses = [];
        girder.restRequest({path: 'HistomicsTK', error: null})
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
        return this;
    },
    _selectAnalysis: function (evt) {
        var name = $(evt.currentTarget).data('name');
        histomicstk.router.setQuery('analysis', name, {trigger: true});
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
    }
});

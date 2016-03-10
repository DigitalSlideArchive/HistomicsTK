histomicstk.App = girder.App.extend({

    initialize: function () {
        girder.fetchCurrentUser()
            .done(_.bind(function (user) {
                girder.eventStream = new girder.EventStream({
                    timeout: girder.sseTimeout || null
                });

                this.headerView = new histomicstk.views.Header({
                    parentView: this
                });

                this.bodyView = new histomicstk.views.Body({
                    parentView: this
                });

                if (user) {
                    girder.currentUser = new girder.models.UserModel(user);
                    girder.eventStream.open();
                }

                this.render();

                Backbone.history.start({pushState: false});
            }, this));

        girder.events.on('g:loginUi', this.loginDialog, this);
        girder.events.on('g:registerUi', this.registerDialog, this);
        girder.events.on('g:resetPasswordUi', this.resetPasswordDialog, this);
        girder.events.on('g:alert', this.alert, this);
        girder.events.on('g:login', this.login, this);
    },
    render: function () {
        this.$el.html(histomicstk.templates.layout());

        this.headerView.setElement(this.$('#g-app-header-container')).render();
        this.bodyView.setElement(this.$('#g-app-body-container')).render();
        return this;
    }
});

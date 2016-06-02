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


        histomicstk.router.on('route', _.bind(function (route, params) {
            var dialog = params.slice(-1)[0].dialog;

            // handle dialog query strings
            if (dialog && _.has(histomicstk.dialogs, dialog)) {
                this.openDialog(dialog);
            } else {
                $('.modal').girderModal('close');
            }
            $('.tooltip').remove();

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
    },
    openDialog: function (name) {
        var dialog = histomicstk.dialogs[name];

        dialog.setElement(this.$('#g-dialog-container'))
            .render();
        dialog.$el.off('hidden.bs.modal')
            .on('hidden.bs.modal', _.bind(this._handleCloseDialog, this));

    },
    _handleCloseDialog: function () {
        histomicstk.router.setQuery('dialog', null, {replace: true});
    }
});

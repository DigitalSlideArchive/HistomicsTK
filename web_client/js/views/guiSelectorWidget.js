histomicstk.views.GuiSelectorWidget = girder.View.extend({
    events: {
        'click a': 'close'
    },

    initialize: function () {
        girder.restRequest({
            path: '/HistomicsTK'
        }).then(_.bind(function (modules) {
            this.modules = modules
            this.render();
        }, this));

    },

    render: function () {
        this.$el.html(
            histomicstk.templates.guiSelectorWidget({
                modules: this.modules || []
            })
        ).girderModal(this);
    },

    close: function () {
        this.$el.modal('hide');
    }
});

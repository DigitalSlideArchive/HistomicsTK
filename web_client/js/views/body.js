histomicstk.views.Body = girder.View.extend({
    initialize: function () {
        this.visView = new histomicstk.views.Visualization({
            parentView: this
        });
        this.panelGroupView = new histomicstk.views.PanelGroup({
            parentView: this
        });
        this.listenTo(histomicstk.router, 'route:main', this.selectGui);
    },
    render: function () {
        this.$el.html(histomicstk.templates.body());
        this.visView.setElement(this.$('#h-vis-container')).render();
        this.panelGroupView.setElement(this.$('#h-panel-group-container')).render();
    },
    selectGui: function () {
        new histomicstk.views.GuiSelectorWidget({
            el: $('#g-dialog-container'),
            parentView: this
        }).render();
    }
});

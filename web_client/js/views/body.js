histomicstk.views.Body = girder.View.extend({
    initialize: function (settings) {
        this.visView = new histomicstk.views.Visualization({
            parentView: this
        });
        // this.panelView = new histomicstk.views.PanelGroupView();
        this.render();
    },
    render: function () {
        this.$el.html(histomicstk.templates.body({}));
        this.visView.setElement(this.$('#h-vis-container')).render();
    }
});

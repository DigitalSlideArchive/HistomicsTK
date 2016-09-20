histomicstk.views.Body = girder.View.extend({
    initialize: function () {
        this.visView = new histomicstk.views.Visualization({
            parentView: this
        });
        this.panelGroupView = new slicer.views.PanelGroup({
            parentView: this
        });
        this.listenTo(histomicstk.events, 'query:analysis', function (analysis) {
            this.panelGroupView.setAnalysis(analysis);
        });
    },
    render: function () {
        this.$el.html(histomicstk.templates.body());
        this.visView.setElement(this.$('#h-vis-container')).render();
        this.panelGroupView.setElement(this.$('#h-panel-controls-container')).render();
    }
});

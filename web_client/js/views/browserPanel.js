histomicstk.views.BrowserPanel = histomicstk.views.Panel.extend({
    render: function () {
        if (!girder.currentUser) {
            this.$el.text('Please login.')
            return;
        }
        this.$el.html(histomicstk.templates.panel(this.spec));
        this._hierarchyView = new girder.views.HierarchyWidget({
            parentView: this,
            parentModel: girder.currentUser,
            checkboxes: false,
            routing: false,
            showActions: false
        });
        this._hierarchyView.setElement(this.$('.h-panel-content')).render();
    }
});

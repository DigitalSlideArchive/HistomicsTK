histomicstk.views.FileSelectorWidget = girder.View.extend({
    initialize: function (settings) {
        this._target = {
            id: settings.id,
            name: settings.name
        };
        this._hierarchyView = new girder.views.HierarchyWidget({
            parentView: this,
            parentModel: girder.currentUser,
            checkboxes: false,
            routing: false,
            showActions: false,
            onItemClick: _.bind(this._selectItem, this)
        });
    },

    render: function () {
        this.$el.html(
            histomicstk.templates.fileSelectorWidget(this._target)
        ).girderModal(this);
        this._hierarchyView.setElement(this.$('.modal-body')).render();
    },

    _selectItem: function (item) {
        this.trigger('g:saved', item);
    }
});

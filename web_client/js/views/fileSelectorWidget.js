histomicstk.views.FileSelectorWidget = girder.View.extend({
    initialize: function (settings) {
        this._target = {
            id: settings.id,
            name: settings.name
        };
    },

    render: function () {
        var hierarchyView = new girder.views.HierarchyWidget({
            parentView: this,
            parentModel: histomicstk.rootPath,
            checkboxes: false,
            routing: false,
            showActions: false,
            onItemClick: _.bind(this._selectItem, this)
        });

        this.$el.html(
            histomicstk.templates.fileSelectorWidget(this._target)
        ).girderModal(this);

        hierarchyView.setElement(this.$('.modal-body')).render();
    },

    _selectItem: function (item) {
        this.trigger('g:saved', item);
    }
});

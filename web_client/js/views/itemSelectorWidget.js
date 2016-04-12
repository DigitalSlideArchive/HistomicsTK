histomicstk.views.ItemSelectorWidget = girder.View.extend({
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
            histomicstk.templates.itemSelectorWidget(this.model.attributes)
        ).girderModal(this);

        hierarchyView.setElement(this.$('.modal-body')).render();
    },

    _selectItem: function (item) {
        this.trigger('g:saved', item);
    }
});

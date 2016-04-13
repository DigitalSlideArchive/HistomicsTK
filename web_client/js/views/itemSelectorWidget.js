histomicstk.views.ItemSelectorWidget = girder.View.extend({
    events: {
        'click .h-select-button': '_selectButton'
    },

    render: function () {
        this._hierarchyView = new girder.views.HierarchyWidget({
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

        this._hierarchyView.setElement(this.$('.h-hierarchy-widget')).render();
    },

    /**
     * Get the currently displayed path in the hierarchy view.
     */
    _path: function () {
        return this._hierarchyView.breadcrumbs.map(function (d) {
            return d.get('name');
        });
    },

    _selectItem: function (item) {
        if (this.model.get('type') === 'file') {
            this.model.set({
                path: this._path(),
                value: item
            });
            this.trigger('g:saved');
        }
    },

    _selectButton: function () {
        var inputEl = this.$('#h-new-file-name');
        var inputElGroup =  inputEl.parent();
        var fileName = inputEl.val() || '';
        var type = this.model.get('type');
        var parent = this._hierarchyView.parentModel;

        inputElGroup.removeClass('has-error');

        switch (type) {
            case 'new-file':

                if (fileName === '') {
                    inputElGroup.addClass('has-error');
                    return;
                }

                this.model.set({
                    path: this._path(),
                    parent: parent,
                    value: new girder.models.ItemModel({
                        name: fileName,
                        folderId: parent.id
                    })
                });
                break;

            case 'directory':
                this.model.set({
                    path: this._path(),
                    value: parent
                });
                break;
        }
        this.trigger('g:saved');
    }
});

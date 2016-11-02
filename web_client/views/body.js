histomicstk.views.Body = girder.View.extend({
    initialize: function () {
        this.visView = new histomicstk.views.Visualization({
            parentView: this
        });
        this.panelGroupView = new slicer.views.PanelGroup({
            parentView: this
        });
        this.listenTo(histomicstk.events, 'query:analysis', function (analysis) {
            if (analysis) {
                this.panelGroupView.setAnalysis(analysis);
            } else {
                this.panelGroupView.reset();
            }
        });
        this.listenTo(
            histomicstk.dialogs.image.model,
            'change',
            function (control) {
                if (!control || !control.get('value') || !control.get('value').id) {
                    return
                }
                this._setImage(control.get('value'));
            }
        );
        this._setImage(histomicstk.dialogs.image.model.get('value'));
    },
    render: function () {
        this.$el.html(histomicstk.templates.body());
        this.visView.setElement(this.$('#h-vis-container')).render();
        this.panelGroupView.setElement(this.$('#h-panel-controls-container')).render();
    },
    /**
     * This loops through all of the models in the control panels and sets any with type
     * "image" to the currently displayed image item.  This is a little hacky.  Ideally,
     * the CLI would provide a special annotation for the main image being processed
     * (whatever that might mean).
     */
    _setImage: function (item) {
        if (!item || !item.get('largeImage')) {
            return;
        }
        var file = new girder.models.FileModel({_id: item.get('largeImage').fileId});
        file.once('g:fetched', function () {
            this.panelGroupView
                .models(undefined, function (m) { return m.get('type') === 'image'})
                .forEach(function (m) {
                    m.set('value', file);
                });
        }, this).fetch();
    }
});

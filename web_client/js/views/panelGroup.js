histomicstk.views.PanelGroup = girder.View.extend({
    events: {
        'click .h-remove-panel': 'removePanel'
    },
    initialize: function () {
        this.panels = [];
        this._panelViews = {};
    },
    render: function () {
        this.$el.html(histomicstk.templates.panelGroup({
            panels: this.panels
        }));
        _.each(this._panelViews, function (view) {
            view.remove();
        });
        _.each(this.panels, _.bind(function (panel) {
            this._panelViews[panel.id] = new histomicstk.views.ControlsPanel({
                parentView: this,
                collection: new histomicstk.collections.Widget(panel.parameters),
                title: panel.label,
                advanced: panel.advanced,
                el: this.$el.find('#' + panel.id)
            });

            this._panelViews[panel.id].render();
        }, this));
    },
    removePanel: function (e) {
        girder.confirm({
            text: 'Are you sure you want to remove this panel?',
            confirmCallback: _.bind(function () {
                var el = $(e.currentTarget).data('target');
                var id = $(el).attr('id');
                this.panels = _.filter(this.panels, function (panel) {
                    return panel.id !== id;
                });
                this.render();
            }, this)
        });
    },

    /**
     * Generate panels from a slicer XML schema.
     * @param {string|XML} s The schema content
     */
    schema: function (s) {
        var gui = histomicstk.schema.parse(s);

        // Create a panel for each "group" in the schema, and copy
        // the advanced property from the parent panel.
        this.panels = _.chain(gui.panels).map(function (panel) {
            return _.map(panel.groups, function (group) {
                group.advanced = !!panel.advanced;
                group.id = _.uniqueId('panel-');
                return group;
            });
        }).flatten(true).value();

        this.render();
    }
});

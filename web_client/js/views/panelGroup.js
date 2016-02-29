histomicstk.views.PanelGroup = girder.View.extend({
    events: {
        'click .h-remove-panel': 'removePanel',
    },
    initialize: function (settings) {
        this.panels = [
            {
                title: 'Item browser',
                type: 'BrowserPanel',
                id: _.uniqueId('panel-')
            }, {
                title: 'Controls',
                type: 'ControlsPanel',
                id: _.uniqueId('panel-')
            }, {
                title: 'Jobs',
                type: 'JobsPanel',
                id: _.uniqueId('panel-')
            }, {
                title: 'Generic panel',
                type: 'Panel',
                id: _.uniqueId('panel-')
            }
        ];
        this._panelViews = {};
    },
    render: function () {
        this.$el.html(histomicstk.templates.panelGroup({
            panels: this.panels
        }));
        _.each(this._panelViews, function (view) {
            view.remove();
        });
        this.panels.forEach(_.bind(function (panel) {
            this._panelViews[panel.id] = new histomicstk.views[panel.type]({
                parentView: this,
                spec: panel,
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
});

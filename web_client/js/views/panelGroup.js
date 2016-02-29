histomicstk.views.PanelGroup = girder.View.extend({
    events: {
        'click .h-remove-panel': 'removePanel',
    },
    initialize: function (settings) {
        this.panels = [
            {
                title: 'Test panel #1',
                type: 'h-test-panel',
                id: _.uniqueId('panel-')
            }, {
                title: 'Test panel #2',
                type: 'h-test-panel',
                id: _.uniqueId('panel-')
            }, {
                title: 'Test panel #3',
                type: 'h-test-panel',
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
            this._panelViews[panel.id] = new histomicstk.views.Panel({
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

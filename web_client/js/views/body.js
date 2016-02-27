histomicstk.views.Body = girder.View.extend({
    events: {
        'click .h-remove-panel': 'removePanel',
        'show.bs.collapse': 'expandPanel',
        'hide.bs.collapse': 'collapsePanel'
    },
    initialize: function (settings) {
        this.visView = new histomicstk.views.Visualization({
            parentView: this
        });
        this.panels = [
            {
                title: 'Test panel',
                class: 'h-test-panel',
                id: _.uniqueId('panel-')
            }
        ];
        this.panelViews = [];

        this.render();
    },
    render: function () {
        this.$el.html(histomicstk.templates.body({
            panels: this.panels
        }));
        this.visView.setElement(this.$('#h-vis-container')).render();
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
    expandPanel: function (e) {
        console.log('expand');
        $(e.currentTarget).find('.icon-down-open').attr('class', 'icon-up-open');
    },
    collapsePanel: function (e) {
        console.log('collapse');
        $(e.currentTarget).find('.icon-up-open').attr('class', 'icon-down-open');
    }
});

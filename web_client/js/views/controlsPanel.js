histomicstk.views.ControlsPanel = histomicstk.views.Panel.extend({
    render: function () {
        this.spec.controls = [
            {
                type: 'range',
                title: 'Select an integer',
                id: 'h-control-integer',
                min: 0,
                max: 100,
                step: 1,
                value: 25
            }, {
                type: 'range',
                title: 'Select a float',
                id: 'h-control-float',
                min: 0,
                max: 1,
                step: 0.01,
                value: 0.25
            }
        ];
        this.$el.html(histomicstk.templates.controlsPanel(this.spec));
        this.$('.h-control-item[data-type="range"] input').slider();
    }
});

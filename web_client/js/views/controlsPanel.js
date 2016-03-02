/* global histomicstk */
histomicstk.views.ControlsPanel = histomicstk.views.Panel.extend({
    render: function () {
        this.spec.controls = [
            {
                type: 'range',
                title: 'Select an integer',
                id: 'h-control-integer-range',
                min: 0,
                max: 100,
                step: 1,
                value: 25
            }, {
                type: 'range',
                title: 'Select a float',
                id: 'h-control-float-range',
                min: 0,
                max: 1,
                step: 0.01,
                value: 0.25
            }, {
                type: 'color',
                title: 'Select a color',
                id: 'h-control-color',
                value: '#00aabb'
            }, {
                type: 'number',
                title: 'Enter an integer',
                id: 'h-control-integer',
                min: 0,
                value: 800
            }, {
                type: 'number',
                title: 'Enter a float',
                id: 'h-control-float',
                min: -10,
                max: 10,
                step: 0.01,
                value: 4.5
            }, {
                type: 'string',
                title: 'Enter a string vector',
                id: 'h-control-string',
                value: 'a string value'
            }, {
                type: 'number-vector',
                title: 'Enter a number vector',
                id: 'h-control-number-vector',
                min: -10,
                max: 10,
                step: 0.01,
                value: [0, 1, 2]
            }, {
                type: 'string-vector',
                title: 'Enter a string vector',
                id: 'h-control-string-vector',
                value: ['value1', 'value2', 'value3']
            }, {
                type: 'enumeration',
                title: 'Select a string',
                id: 'h-control-string-enum',
                values: ['red', 'blue', 'yellow', 'green', 'orange', 'purple'],
                value: 'red',
                numeric: false
            }, {
                type: 'enumeration',
                title: 'Select a number',
                id: 'h-control-numeric-enum',
                values: [10, 5, 40, 100],
                value: 5,
                numeric: true
            }
        ];
        this.$el.html(histomicstk.templates.controlsPanel(this.spec));
        this.$('.h-control-item[data-type="range"] input').slider();
        this.$('.h-control-item[data-type="color"] .input-group').colorpicker({});
    }
});

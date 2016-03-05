histomicstk.views.ControlWidget = Backbone.View.extend({
    events: {},

    initialize: function () {
        this.listenTo(this.model, 'change', this.render);
        this.listenTo(this.model, 'destroy', this.remove);
    },

    render: function () {
        this.$el.html(this.template()(this.model.toJSON()));
        this.$('.h-control-item[data-type="range"] input').slider();
        this.$('.h-control-item[data-type="color"] .input-group').colorpicker({});
        return this;
    },

    /**
     * Type definitions mapping used internally.  Each widget type
     * specifies it's jade template and possibly more custimizations
     * as needed.
     */
    _typedef: {
        range: {
            template: 'rangeWidget'
        },
        color: {
            template: 'colorWidget'
        },
        string: {
            template: 'widget'
        },
        number: {
            template: 'widget'
        },
        boolean: {
            template: 'booleanWidget'
        },
        'string-vector': {
            template: 'vectorWidget'
        },
        'number-vector': {
            template: 'vectorWidget'
        },
        'string-enumeration': {
            template: 'enumerationWidget'
        },
        'number-enumeration': {
            template: 'enumerationWidget'
        },
        file: {
            template: 'fileWidget'
        }
    },

    /**
     * Get the appropriate template for the model type.
     */
    template: function () {
        var type = this.model.get('type');
        var def = this._typedef[type];

        if (def === undefined) {
            console.warn('Invalid widget type "' + type + '"'); // eslint-disable-line no-console
            def = {};
        }
        return histomicstk.templates[def.template] || _.noop;
    }
});

/**
 * A backbone model controlling the behavior and rendering of widgets.
 */
histomicstk.models.Widget = Backbone.Model.extend({
    defaults: {
        type: '',          // The specific widget type
        title: '',         // The label to display with the widget
        value: '',         // The current value of the widget

        // optional attributes only used for certain widget types
        girderModel: {},   // An associate girder model object
        min: 0,            // A minimum value
        max: 1,            // A maximum value
        step: 1,           // Discrete value intervals
        values: []         // A list of possible values
    },


    /**
     * Override Model.set for widget specific bahavior.
     */
    set: function (hash, options) {
        var key, value;

        // handle set(key, value) calling
        if (_.isString(hash)) {
            key = hash;
            value = options;
            options = arguments[2];
            hash = {};
            hash[key] = value;
        }

        return Backbone.Model.prototype.set.call(this, hash, options);
    }
});

histomicstk.collections.Widget = Backbone.Collection.extend({
    model: histomicstk.models.Widget,
    localStorage: new Backbone.LocalStorage('HistomicsTK-Widget-Collection')
});

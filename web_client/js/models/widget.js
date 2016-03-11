/**
 * A backbone model controlling the behavior and rendering of widgets.
 */
histomicstk.models.Widget = Backbone.Model.extend({
    defaults: {
        type: '',          // The specific widget type
        title: '',         // The label to display with the widget
        value: '',         // The current value of the widget

        values: []         // A list of possible values for enum types

        // optional attributes only used for certain widget types
        /*
        girderModel: {},   // An associate girder model object
        min: undefined,    // A minimum value
        max: undefined,    // A maximum value
        step: 1            // Discrete value intervals
        */
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
    },

    /**
     * Coerce a value into a normalized native type.
     */
    normalize: function (value) {
        if (this.isVector()) {
            return this._normalizeVector(value);
        }
        return this._normalizeValue(value);
    },

    /**
     * Coerce a vector of values into normalized native types.
     */
    _normalizeVector: function (value) {
            if (_.isString(value)) {
                value = value.split(',');
            }
            return _.map(value, _.bind(this._normalizeValue, this));
    },

    _normalizeValue: function (value) {
        if (this.isNumeric()) {
            value = parseFloat(value);
        }
        if (this.isBoolean()) {
            value = !!value;
        }
        return value;
    },

    /**
     * Validate the model attributes.
     */
    validate: function (model) {
        if (!_.has(this.types, model.type)) {
            return 'Invalid type, "' + model.type + '"';
        }

        if (this.isVector()) {
            return this._validateVector(model.value);
        }
        return this._validateValue(model.value);
    },

    /**
     * Validate a potential value for the current widget type and properties.
     * This method is called once for each component for vector types.
     */
    _validateValue: function (value) {
        if (this.isNumeric()) {
            return this._validateNumeric(value);
        } else if (this.isEnumeration() && !_.contains(this.get('values'), value.toString())) {
            return 'Invalid valid choice'
        }
        return null;
    },

    /**
     * Validate a potential vector value.  Calls _validateValue internally.
     */
    _validateVector: function (vector) {
        var val;
        vector = this.normalize(vector);
        val = _.chain(3)
            .times(_.bind(this._validateValue, this))
            .reject(_.isNull)
            .value();

        if (val.length === 0) {
            // all components validated
            val = null;
        } else {
            // join errors in individual components
            val = val.join('\n');
        }
        return val;
    },

    /**
     * Validate a numeric value.
     * @param {*} value The value to validate
     * @returns {null|string} An error message or null
     */
    _validateNumeric: function (value) {
        if (!this.isNumeric()) {
            return null;
        }
        var min = parseFloat(this.get('min'));
        var max = parseFloat(this.get('max'));
        var step = parseFloat(this.get('step'));
        var mod, eps = 1e-6;

        value = this.normalize(value);

        // make sure it is a valid number
        if (!Number.isFinite(value)) {
            return 'Invalid number "' + value + '"';
        }

        // make sure it is in valid range
        if (value < min || value > max) {
            return 'Value out of range [' + [min, max] + ']';
        }

        // make sure value is approximately an integer number
        // of "steps" larger than "min"
        min = min || 0;
        mod = (value - min) / step;
        if (step > 0 && Math.abs(Math.round(mod) - mod) > eps) {
            return 'Value does not satisfy step "' + step + '"';
        }
        return null;
    },

    /**
     * True if the value should be coerced as a number.
     */
    isNumeric: function () {
        return _.contains(
            ['range', 'number', 'number-vector', 'number-enumeration'],
            this.get('type')
        );
    },

    /**
     * True if the value should be coerced as a boolean.
     */
    isBoolean: function () {
        return this.get('type') === 'boolean'
    },

    /**
     * True if the value is a 3 component vector.
     */
    isVector: function () {
        return _.contains(
            ['number-vector', 'string-vector'],
            this.get('type')
        );
    },

    /**
     * True if the value should be coerced as a color.
     */
    isColor: function () {
        return this.get('type') === 'color';
    },

    /**
     * True if the value should be chosen from one of several "values".
     */
    isEnumeration: function () {
        return _.contains(
            ['number-enumeration', 'string-enumeration'],
            this.get('type')
        );
    },

    /**
     * True if the value represents a file stored in girder.
     */
    isFile: function () {
        return this.get('type') === 'file';
    }
});

histomicstk.collections.Widget = Backbone.Collection.extend({
    model: histomicstk.models.Widget,
    localStorage: new Backbone.LocalStorage('HistomicsTK-Widget-Collection'),

    /**
     * Get an object containing all of the current parameter values as
     *   modelId -> value
     */
    values: function () {
        var params = {};
        this.each(function (m) {
            params[m.id] = m.get('value');
        });
        return params;
    }
});

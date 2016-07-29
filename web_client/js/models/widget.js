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
        parent: {},        // A parent girder model
        path: [],          // The path of a girder model in a folder hierarchy
        min: undefined,    // A minimum value
        max: undefined,    // A maximum value
        step: 1            // Discrete value intervals
        */
    },

    /**
     * Sets initial model attributes with normalization.
     */
    initialize: function (model) {
        this.set(_.defaults(model || {}, this.defaults));
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

        // normalize values
        if (_.has(hash, 'value')) {
            try {
                hash.value = this.normalize(hash.value);
            } catch (e) {
                console.warn('Could not normalize value "' + hash.value + '"'); // eslint-disable-line no-console
            }
        }

        // normalize enumerated values
        if (_.has(hash, 'values')) {
            try {
                hash.values = _.map(hash.values, _.bind(this.normalize, this));
            } catch (e) {
                console.warn('Could not normalize value in "' + hash.values + '"'); // eslint-disable-line no-console
            }
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
        if (value === '') {
            value = [];
        } else if (_.isString(value)) {
            value = value.split(',');
        }
        return _.map(value, _.bind(this._normalizeValue, this));
    },

    _normalizeValue: function (value) {
        if (this.isNumeric()) {
            value = parseFloat(value);
        } else if (this.isBoolean()) {
            value = !!value;
        } else if (this.isColor()) {
            if (_.isArray(value)) {
                value = {r: value[0], g: value[1], b: value[2]};
            }
            value = tinycolor(value).toHexString();
        } else if (!this.isGirderModel()) {
            value = value.toString();
        }
        return value;
    },

    /**
     * Validate the model attributes.  Returns undefined upon successful validation.
     */
    validate: function (model) {
        if (!_.contains(this.types, model.type)) {
            return 'Invalid type, "' + model.type + '"';
        }

        if (this.isVector()) {
            return this._validateVector(model.value);
        } else if (this.isGirderModel()) {
            return this._validateGirderModel(model);
        }
        return this._validateValue(model.value);
    },

    /**
     * Validate a potential value for the current widget type and properties.
     * This method is called once for each component for vector types.
     */
    _validateValue: function (value) {
        var out;
        if (this.isNumeric()) {
            out = this._validateNumeric(value);
        }
        if (this.isEnumeration() && !_.contains(this.get('values'), this.normalize(value))) {
            out = 'Invalid value choice';
        }
        return out;
    },

    /**
     * Validate a potential vector value.  Calls _validateValue internally.
     */
    _validateVector: function (vector) {
        var val;
        vector = this.normalize(vector);

        val = _.chain(vector)
            .map(_.bind(this._validateValue, this))
            .reject(_.isUndefined)
            .value();

        if (val.length === 0) {
            // all components validated
            val = undefined;
        } else {
            // join errors in individual components
            val = val.join('\n');
        }
        return val;
    },

    /**
     * Validate a numeric value.
     * @param {*} value The value to validate
     * @returns {undefined|string} An error message or null
     */
    _validateNumeric: function (value) {
        var min = parseFloat(this.get('min'));
        var max = parseFloat(this.get('max'));
        var step = parseFloat(this.get('step'));
        var mod, eps = 1e-6;

        // make sure it is a valid number
        if (!isFinite(value)) {
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
    },

    /**
     * Validate a widget that selects a girder model.
     * @note This method is synchronous, so it cannot validate
     * the model on the server.
     */
    _validateGirderModel: function (model) {
        var parent;
        if (!model.value) {
            return 'Empty value';
        }

        switch (this.get('type')) {
            case 'new-file':
                parent = model.parent;
                if (!parent || parent.resourceName !== 'folder') {
                    return 'Invalid parent model';
                }
                break;
            // other model types...
        }
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
        return this.get('type') === 'boolean';
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
     * True if the value represents a model stored in a girder
     * collection/folder/item hierarchy.
     */
    isGirderModel: function () {
        return _.contains(
            ['directory', 'file', 'new-file', 'image'],
            this.get('type')
        );
    },

    /**
     * True if the value represents a file stored in girder.
     */
    isFile: function () {
        return this.get('type') === 'file';
    },

    /**
     * Get a normalized representation of the widget's value.
     */
    value: function () {
        return this.get('value');
    },

    /**
     * A list of valid widget types.
     */
    types: [
        'color',
        'range',
        'number',
        'boolean',
        'string',
        'number-vector',
        'string-vector',
        'number-enumeration',
        'string-enumeration',
        'file',
        'directory',
        'new-file',
        'image'
    ]
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
            // apply special handling for certain parameter types
            // https://github.com/DigitalSlideArchive/HistomicsTK/blob/9e5112ab3444ad8c699d70452a5fe4a74ebbc778/server/__init__.py#L44-L46
            switch (m.get('type')) {
                case 'file':
                    params[m.id + '_girderItemId'] = m.value().id;
                    break;
                case 'new-file':
                    params[m.id + '_girderFolderId'] = m.value().get('folderId');
                    params[m.id + '_name'] = m.value().get('name');
                    break;
                case 'image':
                    params[m.id + '_girderFileId'] = m.value().id;
                    break;
                default:
                    params[m.id] = JSON.stringify(m.value());
            }
        });
        return params;
    }
});

/**
 * This is a parser for Slicer's GUI Schema:
 *   https://www.slicer.org/slicerWiki/index.php/Slicer3:Execution_Model_Documentation#XML_Schema
 */
histomicstk.schema = {
    /**
     * Parse a Slicer GUI spec into a json object for rendering
     * the controlsPanel view.  This function parses into the following structure:
     *
     * * global metadata
     *   * panels[] -- each is rendered in a different panel
     *     * groups[] -- each is rendered together seperated by a horizontal line
     *       * parameters[] -- individual parameters
     *
     * @param {string|object} spec The slicer GUI spec content (accepts parsed or unparsed xml)
     * @returns {object}
     */
    parse: function (spec) {
        var gui, $spec;

        if (_.isString(spec)) {
            spec = $.parseXML(spec);
        }

        $spec = $(spec).find('executable:first');

        // top level metadata
        gui = {
            title: $spec.find('executable > title').text(),
            description: $spec.find('executable > description').text(),
            version: $spec.find('executable > version').text(),
            'documentation-url': $spec.find('executable > documentation-url'),
            license: $spec.find('executable > license').text(),
            contributor: $spec.find('executable > contributor').text(),
            acknowledgements: $spec.find('executable > acknowledgements').text()
        };

        // parameter panels
        gui.panels = _.map($spec.find('executable > parameters'), _.bind(this._parsePanel, this));

        return gui;
    },

    /**
     * Parse a <parameters> tag into a "panel" object.
     */
    _parsePanel: function (panel) {
        var $panel = $(panel);

        return {
            advanced: $panel.attr('advanced') || false,
            groups: _.map($panel.find('> label'), _.bind(this._parseGroup, this))
        };
    },

    /**
     * Parse a parameter group (deliminated by <label> tags) within a
     * panel.
     */
    _parseGroup: function (label) {
        // parameter groups inside panels
        var $label = $(label),
            $description = $label.next('description');

        return {
            label: $label.text(),
            description: $description,
            parameters: _.map($description.next(), _.bind(this._parseParam, this))
        };
    },

    /**
     * Parse an individual parameter element.
     */
    _parseParam: function (param) {
        var type = $(param).get(0).tagName;
        switch (type) {
            case 'integer':
            case 'double':
            case 'boolean':
            case 'string':

                return this._parseScalarParam(type, param);

            case 'integer-vector':
            case 'float-vector':
            case 'double-vector':
            case 'string-vector':

                return this._parseVecterParam(type, param);

            // todo file, directory, image, etc.
        }

        console.warn('Unhandled parameter type "' + type + '"');
        return {}; // todo: filter out invalid params
    },

    /**
     * Parse a scalar parameter type.
     */
    _parseScalarParam: function (type, param) {
        var $param = $(param);
        var widgetTypeMap = {
            integer: 'number',
            float: 'number',
            boolean: 'boolean',
            string: 'string'
        };

        return _.extend(
            {
                type: widgetTypeMap[type],
                slicerType: type,
                id: $param.find('name').text() || $param.find('longflag').text(),
                title: $param.find('label').text(),
                description: $param.find('description').text()
            },
            this._parseDefault(type, $param.find('default')),
            this._parseConstraints($param.find('constraints').get(0))
        );
    },

    /**
     * Parse a `default` tag returning an empty object when no default is given.
     */
    _parseDefault: function (type, value) {
        if (value.length) {
            return {value: value.text()};
        }
        return {};
    },

    /**
     * Parse a `contraints` tag.
     */
    _parseConstraints: function (constraints) {
        var $c = $(constraints);
        var spec = {};
        var min = $c.find('minimum').text();
        var max = $c.find('maximum').text();
        var step = $c.find('step').text();
        if (min) {
            spec.min = min;
        }
        if (max) {
            spec.max = max;
        }
        if (step) {
            spec.step = step;
        }
        return spec;
    }
};

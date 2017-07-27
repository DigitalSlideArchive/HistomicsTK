import _ from 'underscore';

import View from 'girder/views/View';
import girderEvents from 'girder/events';
import { restRequest } from 'girder/rest';
import { confirm } from 'girder/dialog';
import ControlsPanel from 'girder_plugins/item_tasks/views/ControlsPanel';
import WidgetCollection from 'girder_plugins/item_tasks/collections/WidgetCollection';

import events from '../../events';
import JobsPanel from '../../panels/JobsPanel';

import panelGroup from '../../templates/layout/panelGroup.pug';
import '../../stylesheets/layout/panelGroup.styl';

var PanelGroup = View.extend({
    events: {
        'click .h-info-panel-reload': 'reload',
        'click .h-info-panel-submit': 'submit',
        'click .h-remove-panel': 'removePanel'
    },
    initialize: function () {
        this.panels = [];
        this._panelViews = {};

        this._jobsPanelView = new JobsPanel({
            parentView: this,
            spec: {
                title: 'Jobs',
                collapsed: true
            }
        });

        this.listenTo(events, 'query:analysis', this.setAnalysis);
    },
    render: function () {
        this.$el.html(panelGroup({
            info: this._gui,
            panels: this.panels
        }));
        this.$el.addClass('hidden');
        _.each(this._panelViews, function (view) {
            view.remove();
        });
        this._panelViews = {};
        this._jobsPanelView.setElement(this.$('.g-jobs-panel')).render();
        _.each(this.panels, _.bind(function (panel) {
            this.$el.removeClass('hidden');
            this._panelViews[panel.id] = new ControlsPanel({
                parentView: this,
                collection: new WidgetCollection(panel.parameters),
                title: panel.label,
                advanced: panel.advanced,
                el: this.$el.find('#' + panel.id)
            });

            this._panelViews[panel.id].render();
        }, this));

        return this;
    },

    /**
     * Submit the current values to the server.
     */
    submit: function () {
        var params, invalid = false;

        invalid = this.invalidModels();

        if (invalid.length) {
            girderEvents.trigger('g:alert', {
                icon: 'attention',
                text: 'Please enter a valid value for: ' + invalid.map((m) => m.get('title')).join(', '),
                type: 'danger'
            });
            return;
        }

        params = this.parameters();
        _.each(params, function (value, key) {
            if (_.isArray(value)) {
                params[key] = JSON.stringify(value);
            }
        });

        // post the job to the server
        restRequest({
            path: this._submit,
            type: 'POST',
            data: params
        }).then(function (data) {
            events.trigger('h:submit', data);
            return null;
        });
    },

    /**
     * Get the current values of all of the parameters contained in the gui.
     * Returns an object that maps each parameter id to it's value.
     */
    parameters: function () {
        return _.chain(this._panelViews)
            .pluck('collection')
            .invoke('values')
            .reduce(function (a, b) {
                return _.extend(a, b);
            }, {})
            .value();
    },

    /**
     * Return an array of all widget models optionally filtered by the given panel id
     * and model filtering function.
     */
    models: function (panelId, modelFilter) {
        modelFilter = modelFilter || _.constant(true);
        return _.chain(this._panelViews)
            .filter(function (v, i) {
                return panelId === undefined || panelId === i;
            })
            .pluck('collection')
            .pluck('models')
            .flatten()
            .filter(modelFilter)
            .value();
    },

    /**
     * Return an array of all invalid models.  Optionally filter by the given panel id.
     */
    invalidModels: function (panelId) {
        return this.models(panelId, function (m) { return !m.isValid(); });
    },

    /**
     * Return true if all parameters are set and valid.  Also triggers 'invalid'
     * events on each of the invalid models as a byproduct.
     */
    validate: function () {
        return !this.invalidModels().length;
    },

    /**
     * Remove a panel after confirmation from the user.
     */
    removePanel: function (e) {
        confirm({
            text: 'Are you sure you want to remove this panel?',
            confirmCallback: _.bind(function () {
                var el = $(e.currentTarget).data('target');
                var id = $(el).attr('id');
                this.panels = _.reject(this.panels, _.matcher({id: id}));
                this.render();
            }, this)
        });
    },

    /**
     * Remove all panels.
     */
    reset: function () {
        this.panels = [];
        this._gui = null;
        this._submit = null;
        this.render();
    },

    /**
     * Restore all panels to the default state.
     */
    reload: function () {
        if (!this._gui) {
            return this;
        }

        // Create a panel for each "group" in the schema, and copy
        // the advanced property from the parent panel.
        this.panels = _.chain(this._gui.panels).map(function (panel) {
            return _.map(panel.groups, function (group) {
                group.advanced = !!panel.advanced;
                group.id = _.uniqueId('panel-');
                return group;
            });
        }).flatten(true).value();

        this.render();
        return this;
    },

    /**
     * Set the panel group according to the given schema path.
     * This should be a url fragment such as
     *
     *   path = `HistomicsTK/dsarchive_histomicstk_v0.1.3`
     *
     * This code will fetch the actual schema from `path + '/xmlschema'`
     * and cause submissions to post to `path + '/run'`.
     */
    setAnalysis: function (path) {
        if (!path) {
            this.reset();
            return $.when();
        }
        return restRequest({
            path: path + '/xmlspec',
            dataType: 'xml'
        }).then(_.bind(function (xml) {
            this._submit = path + '/run';
            this._schema(xml);
            events.trigger('h:analysis', path, xml);
        }, this));
    },

    /**
     * Generate panels from a json schema.
     */
    _json: function (spec) {
        if (_.isString(spec)) {
            spec = JSON.parse(spec);
        }
        this._gui = spec;
        this.reload();
        return this;
    }
});

export default PanelGroup;

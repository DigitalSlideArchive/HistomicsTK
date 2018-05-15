import $ from 'jquery';
import _ from 'underscore';
import View from 'girder/views/View';

import PluginConfigBreadcrumbWidget from 'girder/views/widgets/PluginConfigBreadcrumbWidget';
import AccessWidget from 'girder/views/widgets/AccessWidget';
import { restRequest } from 'girder/rest';
import events from 'girder/events';
import router from 'girder/router';

import ConfigViewTemplate from '../../templates/body/configView.pug';
import '../../stylesheets/body/configView.styl';

/**
 * Show the default quota settings for users and collections.
 */
var ConfigView = View.extend({
    events: {
        'click #g-histomicstk-save': function (event) {
            this.$('#g-histomicstk-error-message').text('');
            var settings = _.map(this.settingsKeys, (key) => {
                const element = this.$('#g-' + key.replace(/[_.]/g, '-'));
                return {
                    key,
                    value: element.val() || null
                };
            });
            var access = this.accessWidget.getAccessList();
            access.public = this.$('#g-access-public').is(':checked');
            settings.push({key: 'histomicstk.analysis_access', value: access});
            this._saveSettings(settings);
        },
        'click #g-histomicstk-brand-default-color': function () {
            this.$('#g-histomicstk-brand-color').val(this.defaults['histomicstk.brand_color']);
        },
        'click #g-histomicstk-banner-default-color': function () {
            this.$('#g-histomicstk-banner-color').val(this.defaults['histomicstk.banner_color']);
        },
        'click #g-histomicstk-cancel': function (event) {
            router.navigate('plugins', {trigger: true});
        }
    },
    initialize: function () {
        this.breadcrumb = new PluginConfigBreadcrumbWidget({
            pluginName: 'HistomicsTK',
            parentView: this
        });

        this.settingsKeys = [
            'histomicstk.webroot_path',
            'histomicstk.brand_name',
            'histomicstk.brand_color',
            'histomicstk.banner_color',
            'histomicstk.default_draw_styles'
        ];
        $.when(
            restRequest({
                method: 'GET',
                url: 'system/setting',
                data: {
                    list: JSON.stringify(this.settingsKeys),
                    default: 'none'
                }
            }).done((resp) => {
                this.settings = resp;
            }),
            restRequest({
                method: 'GET',
                url: 'system/setting',
                data: {
                    list: JSON.stringify(this.settingsKeys),
                    default: 'default'
                }
            }).done((resp) => {
                this.defaults = resp;
            }),
            restRequest({
                method: 'GET',
                url: 'HistomicsTK/analysis/access'
            }).done((resp) => {
                this.analysisAccess = resp;
            })
        ).done(() => {
            this.render();
        });
    },

    render: function () {
        this.$el.html(ConfigViewTemplate({
            settings: this.settings,
            defaults: this.defaults
        }));
        this.breadcrumb.setElement(this.$('.g-config-breadcrumb-container')).render();
        this.accessWidget = new AccessWidget({
            el: $('#g-histomicstk-analysis-access'),
            modelType: 'Analyses menu',
            model: {
                fetchAccess: () => {
                    return $.Deferred().resolve(this.analysisAccess);
                },
                get: (key) => {
                    if (key === 'public') {
                        return this.analysisAccess.public;
                    } else if (key === 'access') {
                        return this.analysisAccess;
                    }
                }
            },
            hideRecurseOption: true,
            hideSaveButton: true,
            hideAccessType: true,
            noAccessFlag: true,
            modal: false,
            parentView: this
        });
        return this;
    },

    _saveSettings: function (settings) {
        return restRequest({
            method: 'PUT',
            url: 'system/setting',
            data: {
                list: JSON.stringify(settings)
            },
            error: null
        }).done(() => {
            events.trigger('g:alert', {
                icon: 'ok',
                text: 'Settings saved.',
                type: 'success',
                timeout: 4000
            });
        }).fail((resp) => {
            this.$('#g-histomicstk-error-message').text(
                resp.responseJSON.message
            );
        });
    }
});

export default ConfigView;

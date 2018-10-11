import { restRequest } from 'girder/rest';

import events from '../../events';
import router from '../../router';
import View from '../View';

import headerImageTemplate from '../../templates/layout/headerImage.pug';
import '../../stylesheets/layout/headerImage.styl';

var HeaderImageView = View.extend({
    events: {
        'click .h-open-image': function (evt) {
            events.trigger('h:openImageUi');
        },
        'click .h-open-annotated-image': function (evt) {
            events.trigger('h:openAnnotatedImageUi');
        }
    },

    initialize() {
        this.imageModel = null;
        this.parentChain = null;
        this.listenTo(events, 'h:analysis:rendered', this._setNavigationLinks);
        this.listenTo(events, 'h:imageOpened', (model) => {
            this.imageModel = model;
            this.parentChain = null;
            this._setNavigationLinks();
            if (model) {
                this.imageModel.getRootPath((resp) => {
                    this.parentChain = resp;
                    this.render();
                });
            }
            this.render();
        });
    },

    render() {
        this.$el.html(headerImageTemplate({
            image: this.imageModel,
            parentChain: this.parentChain,
            nextImageLink: this.nextImageLink,
            previousImageLink: this.previousImageLink
        }));
        return this;
    },

    _setNavigationLinks() {
        const model = this.imageModel;
        let analysisQuery = '';
        if (!model) {
            this.nextImageLink = null;
            this.previousImageLink = null;
            this.render();
            return;
        }

        if (router.getQuery('analysis')) {
            analysisQuery = `&analysis=${router.getQuery('analysis')}`;
        }
        $.when(
            restRequest({
                url: `item/${model.id}/previous_image`
            }).done((previous) => {
                if (previous._id !== model.id) {
                    this.previousImageLink = `#?image=${previous._id}${analysisQuery}`;
                }
            }),
            restRequest({
                url: `item/${model.id}/next_image`
            }).done((next) => {
                if (next._id !== model.id) {
                    this.nextImageLink = `#?image=${next._id}${analysisQuery}`;
                }
            })
        ).done(() => this.render());
    }
});

export default HeaderImageView;

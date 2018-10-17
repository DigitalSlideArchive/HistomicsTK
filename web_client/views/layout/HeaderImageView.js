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
        this.listenTo(events, 'h:analysis:rendered', this.render);
        this.listenTo(events, 'h:imageOpened', (model) => {
            this.imageModel = model;
            this.parentChain = null;
            this._setNextPreviousImage();
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
        const analysis = router.getQuery('analysis') ? `&analysis=${router.getQuery('analysis')}` : '';
        const nextImageLink = this._nextImage ? `#?image=${this._nextImage}${analysis}` : null;
        const previousImageLink = this._previousImage ? `#?image=${this._previousImage}${analysis}` : null;
        this.$el.html(headerImageTemplate({
            image: this.imageModel,
            parentChain: this.parentChain,
            nextImageLink: nextImageLink,
            previousImageLink: previousImageLink
        }));
        return this;
    },

    _setNextPreviousImage() {
        const model = this.imageModel;
        if (!model) {
            this._nextImage = null;
            this._previousImage = null;
            this.render();
            return;
        }

        $.when(
            restRequest({
                url: `item/${model.id}/previous_image`
            }).done((previous) => {
                if (previous._id !== model.id) {
                    this._previousImage = previous._id;
                }
            }),
            restRequest({
                url: `item/${model.id}/next_image`
            }).done((next) => {
                if (next._id !== model.id) {
                    this._nextImage = next._id;
                }
            })
        ).done(() => this.render());
    }
});

export default HeaderImageView;
